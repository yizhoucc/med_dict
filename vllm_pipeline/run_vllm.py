"""
vLLM Pipeline Runner — standalone extraction pipeline using vLLM API.

This is a simplified version of run.py that uses vLLM HTTP API instead of
direct HuggingFace model loading. It reuses the same prompts, post-processing
hooks, and output format.

Usage:
    1. Start vLLM server: bash vllm_pipeline/start_vllm.sh
    2. Run pipeline: python vllm_pipeline/run_vllm.py exp/v32_vllm.yaml
"""

import sys
import os

# Add parent dir to path so we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import pandas as pd
import json
import time
import re
from datetime import datetime
from typing import Dict

from vllm_pipeline.vllm_client import VLLMClient
from vllm_pipeline.inference import build_base_prompt, vllm_generate
from ult import (
    ChatTemplate,
    try_parse_json,
    extract_schema_keys,
)
from letter_generation import flatten_keypoints, _clean_keypoints_for_letter

# Emotion keywords for letter emotional context (from letter_generation.py)
_EMOTION_KEYWORDS = [
    "distressed", "anxious", "anxiety", "scared", "fearful", "crying", "tearful",
    "depressed", "depression", "worried", "overwhelmed", "upset", "emotional",
    "frightened", "nervous", "stressed",
]
_NEG_WORDS = {'no ', 'not ', 'denies ', 'denied ', 'negative ', 'without ', 'absent ', 'none ',
              'h/o ', 'history of ', 'past medical', 'pmh ', 'hx of ', 'hx '}


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve prompt files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load prompt files
    config["_prompts"] = {}
    prompts_cfg = config.get("prompts", {})
    for key in ["extraction", "plan_extraction", "letter_generation"]:
        path = prompts_cfg.get(key)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                config["_prompts"][key] = yaml.safe_load(f)

    return config


def extract_keypoint(
    task_prompt: str,
    client: VLLMClient,
    gen_config: Dict,
    base_prompt: str,
    chat_tmpl: ChatTemplate,
) -> dict:
    """
    Extract a single keypoint section using vLLM.

    Args:
        task_prompt: The extraction task prompt text
        client: VLLMClient
        gen_config: Generation config
        base_prompt: Base prompt string (system + note)
        chat_tmpl: ChatTemplate instance

    Returns:
        Parsed JSON dict of extracted keypoints
    """
    # Format as a new user turn
    formatted_prompt = chat_tmpl.user_assistant(task_prompt)

    # Generate
    result, _ = vllm_generate(formatted_prompt, client, gen_config, base_prompt)

    # Parse JSON from result
    parsed = try_parse_json(result)
    if parsed is None:
        # Try to fix JSON format
        fix_prompt = chat_tmpl.user_assistant(
            f"The following text should be valid JSON but has errors. "
            f"Fix it and return ONLY the corrected JSON:\n{result}"
        )
        fix_result, _ = vllm_generate(fix_prompt, client, gen_config, base_prompt)
        parsed = try_parse_json(fix_result)

    return parsed if parsed else {}


def extract_assessment_plan_regex(note_text: str) -> str:
    """Extract Assessment/Plan section using regex (same as run.py)."""
    patterns = [
        r'(?:Assessment\s*(?:and|&|/|\\)?\s*Plan|A\s*/\s*P|IMP\s*(?:RESSION)?|ASSESSMENT\s*(?:AND|&)?\s*PLAN|REC(?:OMMENDATIONS)?)\s*[:.]?\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            return note_text[match.start():]
    return ""


def main():
    parser = argparse.ArgumentParser(description="vLLM Pipeline Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    vllm_cfg = model_cfg.get("vllm", {})

    # Create vLLM client
    client = VLLMClient(
        base_url=vllm_cfg.get("base_url", "http://localhost:8000/v1"),
        model_name=model_cfg["name"],
    )

    # Health check
    print("Checking vLLM server...")
    if not client.health_check():
        print("ERROR: vLLM server not reachable. Start it first with: bash vllm_pipeline/start_vllm.sh")
        sys.exit(1)
    print("vLLM server OK")

    # Chat template
    chat_tmpl = ChatTemplate(model_cfg.get("chat_template", "qwen2"))

    # Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["dataset_path"])
    row_indices = data_cfg.get("row_indices")
    if row_indices:
        df = df.iloc[row_indices]
    else:
        row_range = data_cfg.get("row_range", [0, len(df)])
        df = df.iloc[row_range[0]:row_range[1]]
    print(f"Data loaded: {len(df)} rows")

    # Load prompts
    extraction_prompts = config["_prompts"].get("extraction", {})
    plan_extraction_prompts = config["_prompts"].get("plan_extraction", {})
    letter_prompt_template = config["_prompts"].get("letter_generation", {}).get("patient_letter", "")

    # Generation configs
    keypoint_config = config["generation"]["keypoint"].copy()
    retry_config = config["generation"].get("retry", keypoint_config).copy()

    # Setup output directory
    exp_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.txt")

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Starting run: {run_dir}")
    print(f"Model: {model_cfg['name']}")
    print(f"Samples: {len(df)}")
    print()

    total_start = time.time()

    for idx, (row_idx, row) in enumerate(df.iterrows()):
        row_num = row_idx + 1
        note_text = str(row.get("note_text", row.get("text", "")))
        coral_idx = row.get("coral_idx", row_idx)

        print(f"{'='*60}")
        print(f"ROW {row_num} (coral_idx {coral_idx}) [{idx+1}/{len(df)}]")
        row_start = time.time()

        # 1. Build base prompt (replaces build_base_cache)
        base_prompt = build_base_prompt(note_text, chat_tmpl=chat_tmpl)

        # 2. Extract A/P section (regex first)
        assessment_and_plan = extract_assessment_plan_regex(note_text)
        if not assessment_and_plan:
            # LLM fallback
            ap_task = chat_tmpl.user_assistant(
                "Extract the Assessment and Plan section from the note. Return ONLY the A/P text, nothing else."
            )
            assessment_and_plan, _ = vllm_generate(ap_task, client, keypoint_config, base_prompt)

        # 3. Phase 1: Extract keypoints from full note
        keypoints = {}
        phase1_keys = [
            "Reason_for_Visit", "Cancer_Diagnosis", "Lab_Results",
            "Clinical_Findings", "Current_Medications", "Treatment_Changes"
        ]
        for key in phase1_keys:
            prompt = extraction_prompts.get(key, "")
            if not prompt:
                continue
            t0 = time.time()
            result = extract_keypoint(prompt, client, keypoint_config, base_prompt, chat_tmpl)
            keypoints[key] = result
            print(f"  {key}: {time.time()-t0:.1f}s")

        # 4. Phase 2: Extract with context injection
        phase2_keys = ["Treatment_Goals", "Response_Assessment"]
        # Build context from Phase 1 results
        context_parts = []
        for k in ["Cancer_Diagnosis", "Current_Medications", "Clinical_Findings"]:
            if k in keypoints and keypoints[k]:
                context_parts.append(f"{k}: {json.dumps(keypoints[k])}")
        context_str = "\n".join(context_parts)

        for key in phase2_keys:
            prompt = extraction_prompts.get(key, "")
            if not prompt:
                continue
            if context_str:
                prompt = f"Context from earlier extraction:\n{context_str}\n\n{prompt}"
            t0 = time.time()
            # Response_Assessment uses CoT → needs more tokens
            cfg = keypoint_config.copy()
            if key == "Response_Assessment":
                cfg["max_new_tokens"] = 1536
            result = extract_keypoint(prompt, client, cfg, base_prompt, chat_tmpl)
            keypoints[key] = result
            print(f"  {key}: {time.time()-t0:.1f}s")

        # 5. Plan extraction from A/P (Advance_care_planning uses full note)
        if assessment_and_plan:
            ap_base = build_base_prompt(assessment_and_plan, chat_tmpl=chat_tmpl)
            # Fields that need full note context (orders/code status are outside A/P)
            full_note_keys = {"Advance_care_planning", "Imaging_Plan", "Lab_Plan", "Referral", "Genetic_Testing_Plan"}
            for key, prompt in plan_extraction_prompts.items():
                if not prompt:
                    continue
                t0 = time.time()
                ctx = base_prompt if key in full_note_keys else ap_base
                result = extract_keypoint(prompt, client, keypoint_config, ctx, chat_tmpl)
                keypoints[key] = result
                print(f"  {key}: {time.time()-t0:.1f}s")

        # 6. Sanitize + POST hooks
        for section_key, section_val in keypoints.items():
            if isinstance(section_val, dict):
                for field_key, field_val in section_val.items():
                    if isinstance(field_val, list):
                        section_val[field_key] = "; ".join(str(v) for v in field_val)

        # POST hook: lab_summary "Values redacted" for old labs → "No labs in note"
        lab_results = keypoints.get("Lab_Results", {})
        if lab_results.get("lab_summary", "").lower().startswith("values redacted"):
            # Check if note has recent labs (within 6 months)
            # Simple heuristic: if "No visits with results within" appears, labs are old
            if "No visits with results within" in note_text:
                lab_results["lab_summary"] = "No labs in note."
                print("  [POST] lab_summary: old labs → 'No labs in note'")

        # POST hook: imaging_plan empty but note has imaging orders in header
        img_plan = keypoints.get("Imaging_Plan", {})
        if img_plan.get("imaging_plan", "").lower() in ("no imaging planned.", "no imaging planned"):
            # Check for imaging order keywords in note header (before HPI)
            header = note_text.split("History of Present Illness")[0] if "History of Present Illness" in note_text else ""
            img_keywords = ["Bone Scan", "MR Brain", "MRI", "CT ", "PET", "Mammogram", "DEXA", "DXA", "Echocardiogram", "Ultrasound"]
            found = [kw for kw in img_keywords if kw in header]
            if found:
                img_plan["imaging_plan"] = "; ".join(found) + " (ordered in note header)."
                print(f"  [POST] imaging_plan: found header orders → {found}")

        # POST hook: filter oncologic drugs from supportive_meds
        ONCOLOGIC_DRUGS = [
            "letrozole", "tamoxifen", "anastrozole", "exemestane", "fulvestrant",
            "faslodex", "femara", "arimidex", "aromasin",
            "palbociclib", "ibrance", "ribociclib", "kisqali", "abemaciclib", "verzenio",
            "trastuzumab", "herceptin", "pertuzumab", "perjeta", "t-dm1", "kadcyla",
            "everolimus", "afinitor", "capecitabine", "xeloda",
            "goserelin", "zoladex", "leuprolide", "lupron",
        ]
        tx_changes = keypoints.get("Treatment_Changes", {})
        sup_meds = tx_changes.get("supportive_meds", "")
        if sup_meds:
            parts = [p.strip() for p in re.split(r'[;,]', sup_meds)]
            filtered = [p for p in parts if not any(d in p.lower() for d in ONCOLOGIC_DRUGS)]
            if len(filtered) != len(parts):
                tx_changes["supportive_meds"] = "; ".join(filtered) if filtered else ""
                print(f"  [POST] supportive_meds: filtered oncologic drugs")

        # POST hook: medication_plan missing pRBC — check A/P for transfusion
        med_plan = keypoints.get("Medication_Plan", {})
        mp_text = med_plan.get("medication_plan", "").lower()
        if "prbc" not in mp_text and "transfus" not in mp_text and "packed red" not in mp_text:
            ap_text = assessment_and_plan if assessment_and_plan else ""
            ap_lower = ap_text.lower()
            if "prbc" in ap_lower or "packed red" in ap_lower:
                for line in ap_text.split("\n"):
                    ll = line.lower().strip()
                    if "prbc" in ll or "packed red" in ll:
                        med_plan["medication_plan"] = med_plan.get("medication_plan", "") + f" {line.strip()}"
                        print(f"  [POST] medication_plan: added pRBC from A/P")
                        break

        # POST hook: Response_Assessment error → retry with simpler prompt
        resp = keypoints.get("Response_Assessment", {})
        has_error = (resp.get("status") == "error" or
                     "error" in resp or
                     "error" in str(resp.get("message", "")) or
                     (isinstance(resp, dict) and "response_assessment" not in resp and len(resp) <= 2))
        if has_error:
            # Only retry if base_prompt is not too long (avoid exceeding max_model_len)
            if len(base_prompt) < 40000:  # ~10k tokens
                try:
                    simple_prompt = chat_tmpl.user_assistant(
                        'Based on the clinical note, how is the cancer currently responding to treatment? '
                        'Write ONLY a JSON object: {"response_assessment": "your answer"}'
                    )
                    retry_cfg = keypoint_config.copy()
                    retry_cfg["max_new_tokens"] = 512
                    retry_result, _ = vllm_generate(simple_prompt, client, retry_cfg, base_prompt)
                    retry_parsed = try_parse_json(retry_result)
                    if retry_parsed and "response_assessment" in retry_parsed:
                        keypoints["Response_Assessment"] = retry_parsed
                        print(f"  [POST] Response_Assessment: retry succeeded")
                    else:
                        keypoints["Response_Assessment"] = {"response_assessment": "Not mentioned in note."}
                        print(f"  [POST] Response_Assessment: retry failed, using fallback")
                except Exception as e:
                    keypoints["Response_Assessment"] = {"response_assessment": "Not mentioned in note."}
                    print(f"  [POST] Response_Assessment: retry exception: {e}, using fallback")
            else:
                print(f"  [POST] Response_Assessment: note too long for retry, skipping")

        # POST hook: remove "Palliative care" from Specialty if note only discusses palliative INTENT
        referral = keypoints.get("Referral", {})
        spec = referral.get("Specialty", "")
        if "Palliative care" in spec or "palliative care" in spec:
            # Check if note has explicit palliative care REFERRAL (not just intent)
            ap_text = assessment_and_plan or ""
            has_referral = any(p in ap_text.lower() for p in [
                "refer to palliative", "palliative care consult", "palliative consult",
                "referral to palliative", "palliative care referral"
            ])
            if not has_referral:
                parts = [p.strip() for p in spec.split(",")]
                parts = [p for p in parts if "palliative" not in p.lower()]
                referral["Specialty"] = ", ".join(parts) if parts else "None"
                print(f"  [POST] Referral: removed 'Palliative care' (intent, not referral)")

        # POST hook: fix medication_plan contradictions (stopped X but "currently on X")
        DRUG_ALIASES = {
            "fulvestrant": ["faslodex"], "faslodex": ["fulvestrant"],
            "letrozole": ["femara"], "femara": ["letrozole"],
            "anastrozole": ["arimidex"], "arimidex": ["anastrozole"],
            "palbociclib": ["ibrance"], "ibrance": ["palbociclib"],
        }
        mp = keypoints.get("Medication_Plan", {})
        mp_text = mp.get("medication_plan", "")
        mp_lower = mp_text.lower()
        for drug, aliases in DRUG_ALIASES.items():
            all_names = [drug] + aliases
            stopped = any(re.search(rf'stop\w*\s+.*?\b{n}\b', mp_lower) for n in all_names)
            current = any(f"currently on {n}" in mp_lower for n in all_names)
            if stopped and current:
                for n in all_names:
                    mp_text = re.sub(rf'(?i)Currently on {n}\.?\s*', '', mp_text)
                mp["medication_plan"] = mp_text.strip()
                print(f"  [POST] medication_plan: removed contradictory 'currently on {drug}'")

        # POST hook: fix therapy_plan "currently on taxol" when A/P says "taxol planned"
        therapy = keypoints.get("Therapy_plan", {})
        tp_text = therapy.get("therapy_plan", "")
        if "currently on taxol" in tp_text.lower():
            ap_text = assessment_and_plan or ""
            if "taxol planned" in ap_text.lower():
                therapy["therapy_plan"] = tp_text.replace("currently on taxol", "taxol is planned")
                therapy["therapy_plan"] = therapy["therapy_plan"].replace("Currently on taxol", "Taxol is planned")
                print(f"  [POST] therapy_plan: 'currently on taxol' → 'taxol is planned'")

        # --- V33 POST hooks (ported from run.py) ---
        note_lower = note_text.lower()

        # POST-DISTMET-REGIONAL: correct Distant Metastasis if only regional LN sites
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met = cancer.get("Distant Metastasis", "") or ""
            if dist_met and dist_met.lower() not in ("no", "no.", "none", ""):
                dist_lower = dist_met.lower()
                REGIONAL_SITES_DM = ["axillary", "axilla", "sentinel", "supraclavicular",
                                     "infraclavicular", "internal mammary", "chest wall", "ipsilateral"]
                DISTANT_SITES_DM = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                    "ovary", "skin", "adrenal", "contralateral",
                                    "cervical", "distant", "hepatic", "pulmonary",
                                    "osseous", "cerebral", "sternum", "sternal",
                                    "spine", "spinal", "rib", "hip", "femur", "pelvi",
                                    "mediastin", "retroperitoneal", "paraaortic", "para-aortic",
                                    "mesenteric", "inguinal", "scalene"]
                has_regional = any(rs in dist_lower for rs in REGIONAL_SITES_DM)
                has_distant = any(ds in dist_lower for ds in DISTANT_SITES_DM)
                if has_regional and not has_distant:
                    old_dm = cancer["Distant Metastasis"]
                    cancer["Distant Metastasis"] = "No"
                    print(f"  [POST-DISTMET-REGIONAL] '{old_dm}' → 'No' (only regional sites)")

        # POST-STAGE-DISTMET: Stage IV + Distant Met=No → downgrade
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = cancer.get("Stage_of_Cancer", "") or ""
            dist_met = cancer.get("Distant Metastasis", "") or ""
            met = cancer.get("Metastasis", "") or ""
            stage_lower = stage.lower()
            stage_says_iv = "stage iv" in stage_lower or ("metastatic" in stage_lower and "originally" not in stage_lower)
            dist_met_says_no = dist_met.lower().startswith("no")
            if stage_says_iv and dist_met_says_no:
                met_lower = met.lower() if met else ""
                DISTANT_SITES = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                 "ovary", "skin", "contralateral", "distant",
                                 "hepatic", "pulmonary", "osseous", "cerebral",
                                 "sternum", "spine", "rib", "hip", "femur"]
                has_distant = any(ds in met_lower for ds in DISTANT_SITES) if met_lower else False
                if not has_distant:
                    old_stage = stage
                    cleaned = re.sub(r'(?i)\bStage\s*IV\s*\(?\s*metastatic\s*\)?', 'Stage III', stage)
                    cleaned = re.sub(r'(?i)\bmetastatic\s*\(?\s*Stage\s*IV\s*\)?', 'Stage III', cleaned)
                    cleaned = re.sub(r'(?i)\bStage\s*IV\b', 'Stage III', cleaned)
                    cancer["Stage_of_Cancer"] = cleaned
                    print(f"  [POST-STAGE-DISTMET] '{old_stage}' → '{cleaned}' (Distant Met=No)")

        # POST-HER2-VERIFY: HER2- but HER2+ drugs present → correct
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            # Skip if note/extraction explicitly says TNBC/triple-negative (HER2+ drugs may be from prior/mistaken treatment)
            is_tnbc = "triple negative" in type_val.lower() or "tnbc" in type_val.lower()
            note_confirms_tnbc = re.search(r'(?:appears to be|confirmed|is)\s+tnbc|triple[\s-]*negative\s+breast\s+cancer', note_lower)
            # Also skip if metastatic biopsy explicitly shows HER2-negative (HER2+ drugs from original cancer)
            met_biopsy_her2_neg = re.search(r'metastatic\s+biopsy\s+HER2[\s-]*neg', type_val, re.IGNORECASE)
            # Also check note for metastatic biopsy with HER2 neg/1+ AND current non-HER2 therapy
            note_met_bx_her2neg = re.search(
                r'(?:metastatic|recurrence|liver|bone|pelvis)\s+(?:biopsy|bx|fna|ca\s+c/w)[^.]{0,100}(?:\*{3,}[\s-]*1\+|\*{3,}[\s-]*0|\*{3,}[\s-]*neg|her2[\s-]*neg|her[\s-]*2[\s-]*neg)',
                note_lower
            )
            # If current therapy is endocrine-only (letrozole, fulvestrant, etc.), HER2+ drugs are likely from prior treatment
            on_endocrine_only = re.search(r'(?:currently\s+on|on\s+)(?:letrozole|anastrozole|exemestane|fulvestrant|faslodex|tamoxifen)', note_lower)
            if isinstance(type_val, str) and "her2-" in type_val.lower().replace(" ", "") and not is_tnbc and not note_confirms_tnbc and not met_biopsy_her2_neg and not (note_met_bx_her2neg and on_endocrine_only):
                HER2_POS_DRUGS = ["trastuzumab", "pertuzumab", "herceptin", "t-dm1",
                                  "t-dxd", "ado-trastuzumab", "lapatinib", "tykerb", "tucatinib"]
                HER2_POS_REGIMENS = ["tchp", "thp", "ac-thp", "acthp"]
                her2_evidence = []
                for drug in HER2_POS_DRUGS:
                    if drug in note_lower:
                        her2_evidence.append(drug)
                for regimen in HER2_POS_REGIMENS:
                    if re.search(rf'\b{regimen}\b', note_lower):
                        her2_evidence.append(regimen)
                if her2_evidence:
                    old_val = type_val
                    type_val = re.sub(r'(?i)HER2[\s-]*(?:neg(?:ative)?|-)', 'HER2+', type_val)
                    cancer["Type_of_Cancer"] = type_val
                    print(f"  [POST-HER2-VERIFY] '{old_val}' → '{type_val}' (found: {her2_evidence})")

        # POST-GENETICS-SEARCH: Search note for genetic testing when plan says "None"
        gen = keypoints.get("Genetic_Testing_Plan", {})
        if isinstance(gen, dict):
            gen_val = gen.get("genetic_testing_plan", "") or ""
            gen_lower = gen_val.lower().strip()
            if gen_lower in ("none planned.", "none planned", "none", "none.", ""):
                FUTURE_CTX = ["will order", "will send", "send for", "plan to",
                              "interested in", "we will await", "pending",
                              "recommend", "discussed", "consider", "plan for",
                              "will check", "will obtain", "refer for", "schedule",
                              "wishes to", "will likely", "after surgery",
                              "counseling and testing", "counselling and testing"]
                PAST_CTX = ["result:", "results:", "negative for", "positive for",
                            "was done", "already completed", "completed",
                            "will not pursue", "declined", "not interested"]
                GENETIC_TERMS = ["oncotype", "mammaprint", "brca", "genetic counseling",
                                 "genetic counselling", "genetic testing", "molecular profiling",
                                 "germline", "genomic", "foundation one", "foundationone",
                                 "guardant", "tempus", "strata", "invitae", "myriad",
                                 "gene panel", "multigene"]
                found_tests = []
                for term in GENETIC_TERMS:
                    if term not in note_lower:
                        continue
                    for m in re.finditer(re.escape(term), note_lower):
                        start_ctx = max(0, m.start() - 100)
                        end_ctx = min(len(note_lower), m.end() + 100)
                        context = note_lower[start_ctx:end_ctx]
                        if any(pc in context for pc in PAST_CTX):
                            continue
                        if any(fc in context for fc in FUTURE_CTX):
                            found_tests.append(term)
                            break
                # Also check for redacted genomic test pattern: "***** Dx" or "need ***** to determine chemotherapy"
                if not found_tests:
                    genomic_patterns = [
                        r'(?:will\s+(?:likely\s+)?need|plan\s+to\s+(?:get|order|send))\s+\*{3,}.*?(?:chemother|adjuvant)',
                        r'\*{3,}\s*(?:dx|score|result).*?(?:chemother|adjuvant|benefit)',
                        r'(?:await|pending)\s+\*{3,}.*?(?:result|score)',
                    ]
                    for gp in genomic_patterns:
                        if re.search(gp, note_lower):
                            found_tests.append("Genomic test planned (name redacted)")
                            break
                if found_tests:
                    unique = list(dict.fromkeys(found_tests))[:3]
                    gen["genetic_testing_plan"] = ", ".join(unique)
                    print(f"  [POST-GENETICS-SEARCH] Found planned tests: {unique}")

        # POST-IMAGING: Search A/P for imaging keywords (DEXA, echo, etc.)
        img = keypoints.get("Imaging_Plan", {})
        if isinstance(img, dict):
            img_val = img.get("imaging_plan", "") or ""
            img_lower = (img_val or "").lower()
            if img_lower in ("no imaging planned.", "no imaging planned", ""):
                IMAGING_KEYWORDS = {
                    "dexa": "DEXA scan", "bone density": "DEXA scan",
                    "echocardiogram": "Echocardiogram", "echo ": "Echocardiogram",
                    "tte": "Echocardiogram (TTE)",
                    "mammogram": "Mammogram", "ct chest": "CT Chest",
                    "ct abdomen": "CT Abdomen/Pelvis", "ct cap": "CT CAP",
                    "pet/ct": "PET/CT", "pet ct": "PET/CT",
                    "bone scan": "Bone scan", "mri brain": "MRI Brain",
                    "mr brain": "MRI Brain", "mri spine": "MRI Spine",
                }
                FUTURE_IMG = (r'(?:will\s+(?:order|schedule|get|have|obtain|need)|'
                              r'plan\s+(?:for|to)|scheduled?\s+(?:for|a)|'
                              r'consider|recommend|due\s+(?:for|in)|pending|'
                              r'ordered?\s+(?:a\s+)?|need\s+(?:a\s+)?|baseline)')
                ap_lower = (assessment_and_plan or "").lower()
                search_text = ap_lower if ap_lower else note_lower
                found_imgs = []
                for kw, label in IMAGING_KEYWORDS.items():
                    if label.lower() in img_lower:
                        continue
                    pattern = FUTURE_IMG + r'[^.;]{0,40}' + re.escape(kw)
                    alt_pattern = re.escape(kw) + r'\s+(?:due|planned|ordered|scheduled)'
                    if re.search(pattern, search_text, re.IGNORECASE) or \
                       re.search(alt_pattern, search_text, re.IGNORECASE):
                        found_imgs.append(label)
                    elif re.search(r'-\s*' + re.escape(kw), ap_lower):
                        found_imgs.append(label)
                if found_imgs:
                    unique_imgs = list(dict.fromkeys(found_imgs))
                    img["imaging_plan"] = "; ".join(unique_imgs)
                    print(f"  [POST-IMAGING] Found imaging plans: {unique_imgs}")

        # POST-LAB-SEARCH: Search A/P for lab plans when lab_plan says "No labs"
        lab = keypoints.get("Lab_Plan", {})
        if isinstance(lab, dict):
            lab_val = lab.get("lab_plan", "") or ""
            lab_lower = lab_val.lower().strip()
            if lab_lower in ("no labs planned.", "no labs planned", ""):
                LAB_KEYWORDS = {
                    "estradiol": "Estradiol", "fsh": "FSH", "lh": "LH",
                    "check labs": "Labs", "labs monthly": "Monthly labs",
                    "cbc": "CBC", "cmp": "CMP", "tumor marker": "Tumor markers",
                    "ca 15-3": "CA 15-3", "ca 27": "CA 27.29", "cea": "CEA",
                    "hbv dna": "HBV DNA", "hep b": "Hepatitis B monitoring",
                }
                ap_lower = (assessment_and_plan or "").lower()
                found_labs = []
                LAB_FUTURE = r'(?:test(?:ing)?|check|order|monitor|draw|recommend|will\s+(?:order|check|get)|once\s+every|q\s*\d|monthly|every\s+\d)'
                for kw, label in LAB_KEYWORDS.items():
                    if kw in ap_lower:
                        # Check for future context within 60 chars
                        for m in re.finditer(re.escape(kw), ap_lower):
                            ctx = ap_lower[max(0, m.start()-60):m.end()+60]
                            if re.search(LAB_FUTURE, ctx):
                                found_labs.append(label)
                                break
                        # Also match "-DEXA" style bullet points
                    if re.search(r'-\s*' + re.escape(kw), ap_lower):
                        if label not in found_labs:
                            found_labs.append(label)
                if found_labs:
                    unique_labs = list(dict.fromkeys(found_labs))
                    lab["lab_plan"] = "; ".join(unique_labs)
                    print(f"  [POST-LAB-SEARCH] Found lab plans: {unique_labs}")

        # POST-MEDS-SP: Remove completed (s/p) treatments from medication_plan
        med = keypoints.get("Medication_Plan", {})
        if isinstance(med, dict):
            mp = med.get("medication_plan", "") or ""
            if mp and re.search(r'\bs/p\b|status post', mp, re.IGNORECASE):
                sentences = re.split(r'[.;]', mp)
                kept = [s.strip() for s in sentences if s.strip() and
                        not re.search(r'\bs/p\b|status post|completed\s+\d', s, re.IGNORECASE)]
                if kept:
                    med["medication_plan"] = ". ".join(kept) + "."
                else:
                    med["medication_plan"] = "No new medication plans."
                print(f"  [POST-MEDS-SP] Removed s/p (completed) items from medication_plan")

        # POST-MEDS-NOT-STARTED: Remove drugs patient hasn't started from current_meds
        cur_meds = keypoints.get("Current_Medications", {})
        if isinstance(cur_meds, dict):
            meds_val = (cur_meds.get("current_meds", "") or "").strip()
            if meds_val and re.search(r'has not (?:tried|started|begun)', note_lower):
                meds_list = [m.strip() for m in re.split(r'[,;]', meds_val) if m.strip()]
                removed = []
                for med_name in meds_list[:]:
                    med_lower = med_name.lower().split()[0] if med_name else ""
                    if med_lower and re.search(
                        rf'{re.escape(med_lower)}[^.]*?has not (?:tried|started)|'
                        rf'has not (?:tried|started)[^.]*?{re.escape(med_lower)}',
                        note_lower
                    ):
                        meds_list.remove(med_name)
                        removed.append(med_name)
                if removed:
                    cur_meds["current_meds"] = ", ".join(meds_list) if meds_list else ""
                    print(f"  [POST-MEDS-NOT-STARTED] Removed not-yet-started: {removed}")

        # POST-INDETERMINATE-MET: Downgrade distant met if site described as indeterminate
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met = cancer.get("Distant Metastasis", "") or ""
            if "yes" in dist_met.lower():
                INDET_TERMS = ["cyst", "indeterminate", "probably benign", "likely benign",
                               "cannot confirm", "uncertain", "nonspecific"]
                SITES_CHECK = {"liver": ["liver", "hepatic"], "lung": ["lung", "pulmonary"]}
                for site, synonyms in SITES_CHECK.items():
                    if any(s in dist_met.lower() for s in synonyms):
                        for syn in synonyms:
                            for m in re.finditer(re.escape(syn), note_lower):
                                ctx = note_lower[max(0, m.start()-80):m.end()+80]
                                if any(t in ctx for t in INDET_TERMS):
                                    old_dm = dist_met
                                    dist_met = re.sub(
                                        rf'(?i),?\s*(?:and\s+)?(?:to\s+)?{site}[^,;.]*',
                                        '', dist_met
                                    ).strip().rstrip(',')
                                    if not dist_met or dist_met.lower() in ("yes", "yes,", "yes, to"):
                                        dist_met = "No"
                                    cancer["Distant Metastasis"] = dist_met
                                    print(f"  [POST-INDETERMINATE-MET] {site}: indeterminate in note → removed from distant mets")
                                    break

        # POST-BRAIN-STAGING: Remove "brain" from Distant Met if only ordered (not found)
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met = cancer.get("Distant Metastasis", "") or ""
            if "brain" in dist_met.lower() and "suspected" in dist_met.lower():
                # Check if note has actual brain findings (not just "ordered MRI brain")
                brain_findings = re.search(
                    r'(?:brain\s+met|brain\s+lesion|intracranial\s+met|cns\s+(?:met|lesion)|'
                    r'dural\s+enhancement|leptomeningeal|brain.*(?:mass|tumor|nodule))',
                    note_lower
                )
                brain_ordered = re.search(r'(?:order|mri)\s+(?:of\s+)?brain|brain\s+mri', note_lower)
                if brain_ordered and not brain_findings:
                    old_dm = dist_met
                    dist_met = re.sub(r',?\s*(?:and\s+)?(?:to\s+)?brain\s*\(suspected\)', '', dist_met, flags=re.IGNORECASE).strip().rstrip(',')
                    if not dist_met or dist_met.lower() in ("yes", "yes,", "yes, to"):
                        dist_met = "No"
                    cancer["Distant Metastasis"] = dist_met
                    print(f"  [POST-BRAIN-STAGING] Removed 'brain (suspected)' — MRI brain ordered for staging, no findings")

        # POST-LN-CONSISTENCY: Fix conflicting LN counts between findings and Stage
        cancer = keypoints.get("Cancer_Diagnosis", {})
        findings_dict = keypoints.get("Clinical_Findings", {})
        if isinstance(cancer, dict) and isinstance(findings_dict, dict):
            stage_val = cancer.get("Stage_of_Cancer", "") or ""
            findings_val = findings_dict.get("findings", "") or ""
            # Extract LN fractions (X/Y format) from both
            stage_ln = re.findall(r'(\d+)/(\d+)\s*(?:positive\s+)?(?:lymph|LN|node)', stage_val, re.IGNORECASE)
            findings_ln = re.findall(r'(\d+)/(\d+)\s*(?:lymph|LN|node)', findings_val, re.IGNORECASE)
            if stage_ln and findings_ln:
                s_pos, s_total = int(stage_ln[0][0]), int(stage_ln[0][1])
                f_pos, f_total = int(findings_ln[0][0]), int(findings_ln[0][1])
                if s_total == f_total and s_pos != f_pos:
                    # Conflict! Check A/P for the correct value
                    ap_lower = (assessment_and_plan or "").lower()
                    ap_ln = re.findall(r'(\d+)/(\d+)\s*(?:positive\s+)?(?:lymph|ln|node)', ap_lower)
                    if ap_ln:
                        correct_pos = int(ap_ln[0][0])
                        if correct_pos != f_pos:
                            old_findings = findings_val
                            findings_val = re.sub(
                                rf'{f_pos}/{f_total}\s*(?:lymph|LN|node)',
                                f'{correct_pos}/{f_total} lymph node',
                                findings_val
                            )
                            findings_dict["findings"] = findings_val
                            print(f"  [POST-LN-CONSISTENCY] findings LN: {f_pos}/{f_total} → {correct_pos}/{f_total} (from A/P)")

        # POST-LAB-SUMMARY-REDACTED: Fix "Values redacted" when labs are readable
        lab_results = keypoints.get("Lab_Results", {})
        if isinstance(lab_results, dict):
            ls = lab_results.get("lab_summary", "") or ""
            if ls.lower().startswith("values redacted"):
                readable = re.findall(
                    r'(?:Albumin|ALT|AST|Alkaline|WBC|RBC|Hemoglobin|Hematocrit|Platelet|Sodium|Potassium|Creatinine|Glucose|Calcium|Bilirubin)'
                    r'[^*\n]{0,40}\d+\.?\d*',
                    note_text, re.IGNORECASE
                )
                if len(readable) >= 3:
                    lab_results["lab_summary"] = "Labs present in note (see full note for values)."
                    print(f"  [POST-LAB-REDACTED] Overrode 'Values redacted' (found {len(readable)} readable values)")

        # 7. Letter generation
        letter = ""
        letter_metrics = {}
        if letter_prompt_template and config.get("extraction", {}).get("letter", False):
            # 7a. Flatten and clean keypoints (ported from letter_generation.py)
            flat = flatten_keypoints(keypoints)
            flat = _clean_keypoints_for_letter(flat)

            # 7b. Detect patient emotions from note text
            emotions = []
            for kw in _EMOTION_KEYWORDS:
                idx = note_lower.find(kw)
                if idx == -1:
                    continue
                context = note_lower[max(0, idx - 40):idx]
                if any(neg in context for neg in _NEG_WORDS):
                    continue
                emotions.append(kw)
            if emotions:
                flat["emotional_context"] = f"Patient appears {', '.join(emotions[:3])}."

            # 7c. Build letter prompt with flattened keypoints
            keypoints_json = json.dumps(flat, indent=2, ensure_ascii=False)
            letter_prompt_filled = letter_prompt_template.replace("{keypoints_json}", keypoints_json)
            letter_task = chat_tmpl.user_assistant(letter_prompt_filled)
            letter_base = build_base_prompt(note_text, chat_tmpl=chat_tmpl)
            letter_config = keypoint_config.copy()
            letter_config["max_new_tokens"] = 2048
            letter, _ = vllm_generate(letter_task, client, letter_config, letter_base)
            print(f"  Letter: generated ({len(letter)} chars)")

            # POST-LETTER hooks
            # POST-LETTER-METASTASIS: simplify when 3+ organs listed
            MET_ORGANS = ['lung', 'liver', 'bone', 'brain', 'peritoneum', 'ovary', 'ovaries',
                          'pleural', 'adrenal', 'skin', 'spine', 'sternum', 'rib', 'femur',
                          'pelvi', 'hip', 'skull', 'mandible', 'chest wall']
            for m in re.finditer(r'(?:spread|metastasized)\s+to\s+([^.]+)\.', letter, re.IGNORECASE):
                organ_text = m.group(1).lower()
                count = sum(1 for org in MET_ORGANS if org in organ_text)
                if count >= 3:
                    old_sentence = m.group(0)
                    new_sentence = "spread to other parts of your body."
                    letter = letter.replace(old_sentence, new_sentence)
                    print(f"  [POST-LETTER-MET] Simplified {count} organs → 'other parts of your body'")

            # 7d. Compute readability metrics
            try:
                import textstat
                letter_clean = re.sub(r'\[source:[^\]]*\]', '', letter)
                letter_clean = letter_clean.replace('\\n', ' ').replace('\n', ' ')
                letter_clean = re.sub(r'\*\*[^*]+\*\*', '', letter_clean)
                letter_clean = re.sub(r'\s+', ' ', letter_clean).strip()

                if len(letter_clean) > 50:
                    # Raw metrics
                    raw_metrics = {
                        'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(letter_clean), 1),
                        'flesch_reading_ease': round(textstat.flesch_reading_ease(letter_clean), 1),
                        'gunning_fog': round(textstat.gunning_fog(letter_clean), 1),
                        'smog_index': round(textstat.smog_index(letter_clean), 1),
                        'dale_chall': round(textstat.dale_chall_readability_score(letter_clean), 1),
                        'coleman_liau': round(textstat.coleman_liau_index(letter_clean), 1),
                        'ari': round(textstat.automated_readability_index(letter_clean), 1),
                        'linsear_write': round(textstat.linsear_write_formula(letter_clean), 1),
                        'text_standard': textstat.text_standard(letter_clean),
                        'word_count': textstat.lexicon_count(letter_clean),
                        'sentence_count': textstat.sentence_count(letter_clean),
                        'difficult_words': textstat.difficult_words(letter_clean),
                    }

                    # Adjusted metrics (replace explained terms with explanations)
                    letter_adj = re.sub(
                        r'(\b[A-Z][A-Za-z/\-]+(?:\s+[A-Za-z/\-]+){0,2})\s+\((?:a |an |the )?([^)]{5,60})\)',
                        lambda m: m.group(2) if any(c.isalpha() for c in m.group(2)) and not any(d in m.group(2) for d in ['mg', 'U/L', 'g/dL', 'mmol', 'IHC', 'FISH', 'redacted']) else m.group(0),
                        letter_clean
                    )
                    adj_metrics = {
                        'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(letter_adj), 1),
                        'flesch_reading_ease': round(textstat.flesch_reading_ease(letter_adj), 1),
                        'dale_chall': round(textstat.dale_chall_readability_score(letter_adj), 1),
                        'difficult_words': textstat.difficult_words(letter_adj),
                    }

                    letter_metrics = {'raw': raw_metrics, 'adjusted': adj_metrics}
                    fk = raw_metrics['flesch_kincaid_grade']
                    fk_adj = adj_metrics['flesch_kincaid_grade']
                    fre = raw_metrics['flesch_reading_ease']
                    print(f"  Readability: FK={fk} (adj={fk_adj}), FRE={fre}, words={raw_metrics['word_count']}, {raw_metrics['text_standard']}")
            except ImportError:
                pass  # textstat not installed, skip metrics

        # 8. Write results
        row_time = time.time() - row_start
        with open(results_path, 'a') as f:
            f.write(f"RESULTS FOR ROW {row_num}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"--- Column: coral_idx ---\n{coral_idx}\n\n")
            f.write(f"--- Column: note_text ---\n{json.dumps(note_text)}\n\n")
            f.write(f"--- Column: assessment_and_plan ---\n{json.dumps(assessment_and_plan)}\n\n")
            f.write(f"--- Column: keypoints ---\n{json.dumps(keypoints, indent=2)}\n\n")
            f.write(f"--- Column: letter ---\n{json.dumps(letter)}\n\n")
            if letter_metrics:
                f.write(f"--- Column: readability_metrics ---\n{json.dumps(letter_metrics, indent=2)}\n\n")
            f.write("\n\n\n\n\n")

        print(f"  Row {row_num} total: {row_time:.1f}s")
        print()

    total_time = time.time() - total_start
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results in: {run_dir}/")


if __name__ == "__main__":
    main()
