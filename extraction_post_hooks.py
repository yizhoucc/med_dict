"""Pure, dependency-free helpers for conservative oncology extraction post-hooks."""

import re


M1_SITE_RE = re.compile(
    r"liver|hepatic|lung|pulmonary|bone|osseous|brain|cerebral|pleur\w*|peritone\w*|"
    r"adrenal|spine|vertebr\w*|omentum|omental|abdominal[\s-]*wall|contralateral|"
    r"cervical\s+(?:lymph\s*)?nodes?",
    re.IGNORECASE,
)

UNCERTAINTY_RE = re.compile(
    r"\b(?:not sure|unsure|suspect\w*|suspicious|possible|uncertain|equivocal|pending|"
    r"cannot exclude|concern\w*|indeterminate)\b",
    re.IGNORECASE,
)


def _other_primary_terms(cancer_type):
    if cancer_type == "breast":
        return (
            "ovarian", "endometrial", "colon cancer", "colorectal", "lung cancer",
            "thyroid cancer", "melanoma", "pancreatic", "prostate cancer", "renal cell",
        )
    return (
        "breast cancer", "mammary", "ovarian", "endometrial", "colon cancer",
        "colorectal", "prostate cancer", "melanoma", "lung cancer", "thyroid cancer",
        "renal cell",
    )


def _sentence_bounds(text, start, end):
    sentence_start = max(text.rfind(sep, 0, start) for sep in (".", ";", "\n")) + 1
    sentence_ends = [text.find(sep, end) for sep in (".", ";", "\n")]
    sentence_ends = [value for value in sentence_ends if value >= 0]
    sentence_end = min(sentence_ends) if sentence_ends else len(text)
    return sentence_start, sentence_end


def has_affirmative_m1_site(text, cancer_type):
    """Return True only for a non-negated, non-benign M1 site with a malignant anchor."""
    text = str(text or "").lower()
    other_primary_terms = _other_primary_terms(cancer_type)
    for match in M1_SITE_RE.finditer(text):
        clause_start, clause_end = _sentence_bounds(text, match.start(), match.end())
        clause = text[clause_start:clause_end]
        pre = text[clause_start:match.start()]
        post = text[match.end():clause_end]
        if any(term in clause for term in other_primary_terms):
            continue
        if re.search(
            r"\b(?:no|not|without|negative|suspect\w*|possible|uncertain|equivocal|pending|"
            r"cannot exclude|concern\w*|indeterminate|history of|historical|previously|prior|"
            r"originally|if|whether)\b",
            pre,
        ):
            continue
        if re.search(r"\b(?:benign|cyst\w*|hemangioma|resected|resolved|ned)\b", clause):
            continue
        if UNCERTAINTY_RE.search(clause):
            site_term = re.escape(match.group(0))
            explicitly_confirmed_site = bool(
                re.search(rf"\b(?:confirmed|biopsy[\s-]*(?:proven|confirmed))\b[^.;]{{0,35}}{site_term}", clause)
                or re.search(rf"{site_term}\s+(?:metasta\w*|malignan\w*|carcinoma\w*)\b", clause)
            )
            if not explicitly_confirmed_site:
                continue
        if re.search(
            r"\b(?:yes|confirmed|metasta\w*|malignan\w*|carcinoma\w*|stage\s*iv)\b",
            clause,
        ):
            return True
    return False


def has_uncertain_m1_site(text, cancer_type):
    """Return True for a current nonregional site that is explicitly uncertain."""
    text = str(text or "").lower()
    other_primary_terms = _other_primary_terms(cancer_type)
    for match in M1_SITE_RE.finditer(text):
        sentence_start, sentence_end = _sentence_bounds(text, match.start(), match.end())
        context = text[sentence_start:sentence_end]
        pre = text[max(sentence_start, match.start() - 50):match.start()]
        if any(term in context for term in other_primary_terms):
            continue
        if re.search(
            r"\bno\b|\bnot\s+(?!sure\b)|\b(?:without|negative|history of|historical|"
            r"previously|prior|originally|resected|resolved)\b",
            pre,
        ):
            continue
        if re.search(r"\b(?:benign|cyst\w*|hemangioma|resected|resolved|ned)\b", context):
            continue
        if UNCERTAINTY_RE.search(context):
            return True
    return False


def is_confirmed_distant_value(value, cancer_type):
    """Interpret a Distant Metastasis value without flattening mixed site certainty."""
    value = str(value or "").lower().strip()
    explicitly_negative = bool(
        re.search(r"\bno\s+(?:evidence of\s+)?distant\b|\bnegative for\s+distant\b", value)
    )
    if not value.startswith("yes") or explicitly_negative:
        return False
    if value.rstrip(" .,;:") == "yes":
        return True
    if has_affirmative_m1_site(value, cancer_type):
        return True
    return bool(re.search(r"\b(?:confirmed\s+)?distant\s+(?:metasta\w*|disease)\b", value))


def has_explicit_m1_evidence(text, cancer_type):
    """Detect current, explicit M1 evidence while excluding negated/history/other-primary text."""
    text = str(text or "").lower()
    other_primary_terms = _other_primary_terms(cancer_type)
    pattern = re.compile(
        r"\bstage\s*(?:iv|4)\b|\bde novo mbc\b|"
        r"\bmetastatic\s+(?:breast\s+cancer|pancreatic\s+cancer|pdac)\b|"
        r"peritoneal carcinomatosis|omental cak(?:e|ing)|"
        r"metastatic[^.]{0,35}(?:to|in)\s+(?:the\s+)?"
        r"(?:liver|hepatic|lung|pulmonary|bone|osseous|brain|cerebral|pleur|peritone|"
        r"adrenal|spine|vertebr|abdominal[\s-]*wall|cervical\s+(?:lymph\s*)?node)|"
        r"(?:biopsy[\s-]*(?:proven|confirmed)|confirmed)[^.]{0,45}"
        r"(?:liver|hepatic|lung|pulmonary|bone|osseous|brain|cerebral|pleur|peritone|"
        r"adrenal|spine|vertebr|abdominal[\s-]*wall|cervical\s+(?:lymph\s*)?node)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        pre = text[max(0, match.start() - 45):match.start()]
        sentence_start, sentence_end = _sentence_bounds(text, match.start(), match.end())
        context = text[sentence_start:sentence_end]
        if re.search(
            r"\b(?:no|not|without|negative for|rule out|r/o|suspect\w*|suspicious|possible|"
            r"possibility|concern\w*|pending|presumptiv\w*|cannot exclude|if|whether|would|could)\b",
            pre,
        ):
            continue
        if re.search(r"\b(?:if|unless|pending confirmation)\b", context):
            continue
        if any(term in context for term in other_primary_terms):
            continue
        if re.search(
            r"\b(?:history of|historical|previously|prior|originally|status post|s/p|resected)\b|"
            r"\b(?:currently\s+)?(?:ned|no evidence of disease)\b",
            context,
        ):
            continue
        return True
    return False


def met_status(value):
    value = str(value or "").lower().strip()
    if not value:
        return "EMPTY"
    uncertain = bool(UNCERTAINTY_RE.search(value))
    affirmative = "yes" in value or "confirmed" in value
    if affirmative and uncertain:
        return "MIXED"
    if uncertain:
        return "UNSURE"
    if affirmative:
        return "YES"
    if value in ("no", "no.", "none") or value.startswith(("no ", "no,", "no.")):
        return "NO"
    return "OTHER"


def reconcile_metastasis_fields(distant, general, cancer_type):
    """Reconcile only safe subset/certainty cases; never flatten a mixed regional+distant value."""
    distant = str(distant or "").strip()
    general = str(general or "").strip()
    distant_status = met_status(distant)
    general_status = met_status(general)
    distant_confirmed = is_confirmed_distant_value(distant, cancer_type)
    has_distant = has_affirmative_m1_site(general, cancer_type)
    mentions_distant = bool(M1_SITE_RE.search(general))
    regional_scan = re.sub(
        r"(?:contralateral(?:\s+(?:right|left))?\s+(?:axill\w*|supraclavicular)|"
        r"cervical)(?:\s+(?:lymph\s*)?nodes?)?",
        "",
        general.lower(),
    )
    has_regional = any(
        term in regional_scan
        for term in (
            "axill", "sentinel", "subpectoral", "supraclavicular", "infraclavicular",
            "internal mammary", "chest wall", "ipsilateral", "regional", "lymph node", "nodal",
        )
    )
    reason = None
    if distant_confirmed and general_status in ("EMPTY", "NO"):
        general = distant
        reason = "confirmed distant→general"
    elif general_status == "YES" and has_distant and not has_regional and distant_status == "EMPTY":
        sites = re.sub(
            r"(?i)^\s*(?:yes|suspected|not sure)?[\s,:.()-]*(?:to\s+)?", "", general
        ).strip().strip("()").strip()
        suspected = f"Suspected, to {sites}" if sites and sites.lower() != "yes" else "Not sure"
        distant = general = suspected
        reason = "unconfirmed distant-only general claim"
    elif distant_status == "NO" and general_status == "YES" and has_distant and not has_regional:
        general = "No"
        reason = "Distant=No rejects distant-only general claim"
    elif distant_status == "UNSURE" and general_status in ("EMPTY", "NO"):
        general = distant
        reason = "uncertain distant→empty general"
    elif distant_status == "UNSURE" and general_status == "YES" and has_distant and not has_regional:
        general = distant
        reason = "distant certainty ceiling"
    elif (
        distant_status == "UNSURE"
        and general_status == "UNSURE"
        and distant != general
        and (
            general.lower() in ("not sure", "unsure", "suspected", "possible")
            or (not mentions_distant and M1_SITE_RE.search(distant))
        )
    ):
        general = distant
        reason = "preserve uncertain distant site detail"
    elif general_status == "UNSURE" and distant_status == "EMPTY" and mentions_distant:
        distant = general
        reason = "explicit uncertain M1 site→Distant"
    return distant, general, reason


def clean_breast_distant(value):
    """Remove explicitly regional breast sites while preserving contralateral M1 nodes."""
    value = str(value or "")
    lower = value.lower()
    contralateral_pattern = (
        r"contralateral(?:\s+(?:right|left))?\s+"
        r"(?:axillary|axilla|supraclavicular)(?:\s+(?:lymph\s*)?nodes?)?"
    )
    regional_scan = re.sub(contralateral_pattern, "", lower, flags=re.IGNORECASE)
    regional_sites = (
        "axillary", "axilla", "sentinel", "subpectoral", "supraclavicular",
        "infraclavicular", "internal mammary", "chest wall", "ipsilateral",
    )
    has_regional = any(site in regional_scan for site in regional_sites)
    generic_node_suffix = bool(
        re.search(r"(?:\s+and\s+|,\s*)(?:regional\s+)?(?:lymph\s*)?nodes?\s*$", value, re.IGNORECASE)
    )
    if not (has_regional or generic_node_suffix):
        return value

    protected = []

    def protect(match):
        protected.append(match.group(0))
        return f"__M1NODE_{len(protected) - 1}__"

    cleaned = re.sub(contralateral_pattern, protect, value, flags=re.IGNORECASE)
    regional_phrase = (
        r"(?:right|left|ipsilateral)?\s*(?:axillary|axilla|sentinel|subpectoral|"
        r"supraclavicular|infraclavicular|internal mammary|chest[\s-]*wall)"
        r"(?:\s+(?:lymph\s*)?nodes?)?"
    )
    cleaned = re.sub(
        rf"(?:\s*[,;/]\s*|\s+and\s+|\s+to\s+)?{regional_phrase}",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:and|,)\s+(?:regional\s+)?(?:lymph\s*)?nodes?\s*$", "", cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*([,;])\s*$", "", cleaned).strip()
    for index, site in enumerate(protected):
        cleaned = cleaned.replace(f"__M1NODE_{index}__", site)
    cleaned = re.sub(r"(?i)^\s*yes\s*[,;:]?\s*(?:and\s+)", "Yes, ", cleaned)
    cleaned = re.sub(r"(?i)^\s*yes\s*,?\s*;\s*", "Yes, ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")

    # A bare word such as "distant" inside a negated clause is not an M1 anchor.
    # Keep only an affirmative M1 claim, or a genuinely uncertain nonregional site.
    if cleaned.lower().rstrip(" .,;:") == "yes":
        return "No"
    if is_confirmed_distant_value(cleaned, "breast"):
        return cleaned
    uncertain_site = has_uncertain_m1_site(cleaned, "breast")
    negated_distant = bool(
        re.search(r"\bno\s+(?:evidence of\s+)?distant\b|\bno\s+distant\s+disease\b", cleaned, re.I)
    )
    if uncertain_site and not negated_distant:
        return cleaned
    return "No"


def regional_node_evidence(cancer_type, stage, assessment_and_plan):
    """Return only high-confidence regional evidence: p/ypN+ or current A/P count/biopsy."""
    stage = str(stage or "")
    text = str(assessment_and_plan or "").lower()
    path_tnm = re.search(
        r"\b((?:yp|p)T\d[a-d]?(?:\([^)]*\))?\s*,?\s*(?:yp|p)?N([1-3])([a-c]?))",
        stage,
        re.IGNORECASE,
    )
    standalone_pn = re.search(r"\b((?:yp|p)N([1-3])([a-c]?))\b", stage, re.IGNORECASE)
    if path_tnm or standalone_pn:
        match = path_tnm or standalone_pn
        detail = re.search(r"(?:yp|p)?N[1-3][a-c]?", match.group(1), re.IGNORECASE).group(0)
        return "HISTORICAL", detail

    current_terms = (("breast", "mammary") if cancer_type == "breast"
                     else ("pancrea", "pdac", "whipple", "pancreaticoduoden"))
    other_terms = _other_primary_terms(cancer_type)

    def context_allowed(match):
        sentence_start, sentence_end = _sentence_bounds(text, match.start(), match.end())
        context = text[sentence_start:sentence_end]
        pre = text[max(sentence_start, match.start() - 55):match.start()]
        if any(term in context for term in other_terms) or not any(term in context for term in current_terms):
            return False
        if cancer_type == "breast":
            node_context = text[
                max(sentence_start, match.start() - 30):min(sentence_end, match.end() + 30)
            ]
            if "contralateral" in node_context:
                return False
        if re.search(
            r"\b(?:history of|historical|previously|prior|status post|s/p|in 20\d{2})\b", context
        ):
            return False
        if re.search(
            r"\b(?:if|whether|should|would)\b[^.;\n]{0,45}$|patients?\s+with[^.;\n]{0,30}$", pre
        ):
            return False
        if re.search(
            r"negative for (?:carcinoma|malignancy|metastasis)|no (?:evidence of )?"
            r"(?:nodal|lymph[\s-]*node) (?:disease|involvement|metastasis)|benign|reactive|"
            r"granulomatous|sarcoid",
            match.group(0),
        ):
            return False
        return True

    for match in re.finditer(
        r"\b(\d+)\s*(?:/|of)\s*(\d+)\s*(?:regional\s+)?(?:lymph\s*)?"
        r"(?:nodes?|lns?|slns?)\b[^.\n;]{0,45}\b(?:positive|involved|metastatic|"
        r"with\s+(?:micro|macro)?metasta)",
        text,
    ):
        if int(match.group(1)) > 0 and context_allowed(match):
            return "CONFIRMED", f"{match.group(1)}/{match.group(2)} nodes positive"

    if cancer_type == "breast":
        breast_nodes = (
            r"(?:axillary|axilla|sentinel|subpectoral|supraclavicular|infraclavicular|"
            r"internal mammary)(?:\s+(?:(?:lymph\s*)?nodes?|lns?))"
        )
        patterns = (
            rf"(?:fna|fine needle aspiration|core biopsy|biopsy)[^.\n]{{0,100}}{breast_nodes}"
            rf"[^.\n]{{0,80}}(?:metastatic|positive for|involved by|malignan|adenocarc|carcinoma)",
            rf"{breast_nodes}[^.\n]{{0,80}}(?:fna|fine needle aspiration|core biopsy|biopsy|"
            rf"positive for|involved by|contains)[^.\n]{{0,80}}"
            rf"(?:metastatic|malignan|adenocarc|carcinoma)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match and context_allowed(match):
                site_match = re.search(breast_nodes, match.group(0))
                detail = site_match.group(0) if site_match else "regional node pathology positive"
                return "CONFIRMED", detail
    return None, ""


def compose_general_metastasis(distant, evidence, detail, cancer_type):
    prefix = "historical pathologically confirmed" if evidence == "HISTORICAL" else "confirmed"
    suffix = f" ({detail})" if detail else ""
    regional_clause = f"{prefix} regional lymph-node involvement{suffix}"
    status = met_status(distant)
    if is_confirmed_distant_value(distant, cancer_type):
        return f"{distant}; {regional_clause}"
    if status in ("UNSURE", "MIXED"):
        return f"Yes, {regional_clause}; distant disease uncertain — {distant}"
    if status == "NO":
        return f"Yes, {regional_clause}; no distant metastasis"
    return f"Yes, {regional_clause}"


def merge_regional_metastasis(distant, general, evidence, detail, cancer_type):
    """Append confirmed regional disease without losing the Distant field's site certainty."""
    distant = str(distant or "").strip()
    general = str(general or "").strip()
    if not evidence:
        return general, False

    regional_scan = re.sub(
        r"(?:contralateral(?:\s+(?:right|left))?\s+(?:axill\w*|supraclavicular)|"
        r"cervical)(?:\s+(?:lymph\s*)?nodes?)?",
        "",
        general.lower(),
    )
    has_regional = bool(
        re.search(
            r"regional|axill|sentinel|subpectoral|supraclavicular|infraclavicular|"
            r"internal mammary|lymph[\s-]*node|\bnodal\b",
            regional_scan,
        )
    )
    distant_status = met_status(distant)
    if is_confirmed_distant_value(distant, cancer_type):
        has_matching_distant_state = is_confirmed_distant_value(general, cancer_type)
    elif distant_status in ("UNSURE", "MIXED"):
        has_matching_distant_state = bool(
            UNCERTAINTY_RE.search(general) and M1_SITE_RE.search(general)
        )
    else:
        has_matching_distant_state = True

    general_status = met_status(general)
    needs_update = (
        general_status in ("EMPTY", "NO", "UNSURE")
        or not has_regional
        or not has_matching_distant_state
    )
    if not needs_update:
        return general, False
    return compose_general_metastasis(distant, evidence, detail, cancer_type), True


def normalize_stage_iv(stage, distant, assessment_and_plan, cancer_type):
    """Downgrade only unsupported *confirmed* Stage-IV assertions; preserve suspected M1."""
    stage = str(stage or "")
    stage_lower = stage.lower()
    if not re.search(r"stage\s*iv|metastatic", stage_lower):
        return stage, None
    if any(word in stage_lower for word in ("suspect", "possible", "pending", "presumpt")):
        return stage, None
    if is_confirmed_distant_value(distant, cancer_type) or has_explicit_m1_evidence(
        assessment_and_plan, cancer_type
    ):
        return stage, None
    if has_uncertain_m1_site(distant, cancer_type) or has_uncertain_m1_site(
        assessment_and_plan, cancer_type
    ):
        return "Suspected Stage IV (pending confirmation)", "unconfirmed M1 evidence"
    return "Not staged in note", "no confirmed M1 basis"


def verify_unique_pathologic_tnm(stage, note_text):
    """Correct an extracted p/ypTN only when the note has one unique formal pathology TN."""
    stage = str(stage or "")
    value_tnm = re.search(r"(?:yp|p)T\s*\d[a-d]?\s*N\s*\d[a-c]?", stage, re.IGNORECASE)
    if not value_tnm:
        return stage
    formal_pattern = re.compile(
        r"(?:AJCC[^.\n]{0,100})?Pathologic\s+Stage(?:\s+Classification)?\s*:?\s*"
        r"\b((?:yp|p)T\d[a-d]?\s*,?\s*N\d[a-c]?)",
        re.IGNORECASE,
    )
    candidates = {}
    for match in formal_pattern.finditer(str(note_text or "")):
        normalized = re.sub(r"\s|,", "", match.group(1)).lower()
        prefix_match = re.match(r"(yp|p)t(\d)([a-d]?)n(\d)([a-c]?)", normalized)
        if prefix_match:
            canonical = (
                f"{prefix_match.group(1)}T{prefix_match.group(2)}{prefix_match.group(3)}"
                f"N{prefix_match.group(4)}{prefix_match.group(5)}"
            )
            candidates[normalized] = canonical
    if len(candidates) != 1:
        return stage
    normalized_note_tnm, note_tnm = next(iter(candidates.items()))
    if re.sub(r"\s|,", "", value_tnm.group(0)).lower() == normalized_note_tnm:
        return stage
    return re.sub(
        r"(?:yp|p)T\s*\d[a-d]?\s*N\s*\d[a-c]?",
        note_tnm,
        stage,
        count=1,
        flags=re.IGNORECASE,
    )


def locally_advanced_stage(text, cancer_type):
    """Return a descriptive extent label; numeric Stage III requires an explicit source statement."""
    text = str(text or "").lower()
    explicit_stage_iii = bool(re.search(r"\bstage\s*(?:iii|3)\b", text))
    patient_locally_advanced = False
    for match in re.finditer(r"locally[- ]advanced", text):
        pre = text[max(0, match.start() - 14):match.start()]
        if not re.search(r"\bor\s+$|patients with[^.]*$", pre):
            patient_locally_advanced = True
            break
    vessel = cancer_type != "breast" and bool(
        re.search(
            r"(?:encase|encasement|occlu|abut|>?\s*180)[^.]{0,40}"
            r"(?:sma\b|superior mesenteric|celiac|splenic (?:artery|vein)|portal vein|smv\b|"
            r"hepatic artery|mesenteric)|"
            r"(?:sma\b|superior mesenteric|celiac|splenic (?:artery|vein)|portal vein|smv\b|"
            r"hepatic artery)[^.]{0,40}(?:encase|encasement|occlu|abut|>\s*180|"
            r"contact greater than 180)",
            text,
        )
    )
    if not (patient_locally_advanced or vessel):
        return None
    if explicit_stage_iii:
        return "Stage III (locally advanced)"
    if cancer_type != "breast" and ("unresectable" in text or vessel):
        return "Locally advanced (unresectable)"
    return "Locally advanced"
