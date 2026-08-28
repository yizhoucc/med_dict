"""Pure, dependency-free helpers for conservative oncology extraction post-hooks."""

import re


M1_SITE_RE = re.compile(
    r"liver|hepatic|lung|pulmonary|bone|osseous|brain|cerebral|pleur\w*|peritone\w*|"
    r"adrenal|spine|vertebr\w*|ilium|iliac|sacrum|sacral|omentum|omental|"
    r"abdominal[\s-]*wall|contralateral|"
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
    if re.search(r"^yes\b[^.;]*\bhistorically confirmed to\b", value):
        return True
    # Pancreatic cancer can recur in explicitly nonregional abdominal nodal basins.
    # These values have already passed the dedicated Distant-field sanitizer, so a
    # plain affirmative result here should remain confirmed for Stage reconciliation.
    if cancer_type == "pdac" and re.search(
        r"^yes\b[^.;]*(?:intra[ -]?abdominal|mesenteric|gastrohepatic|"
        r"retroperitoneal|aortocaval|periaortic)\s+(?:lymph\s*)?nodes?\b",
        value,
        re.IGNORECASE,
    ):
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
        r"\bmetastatic\s+(?:adenocarcinoma|carcinoma)\s+of\s+(?:the\s+)?(?:breast|pancreas)\b|"
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
        # A proposed biopsy/work-up "for a definitive Stage IV diagnosis" means the M1
        # diagnosis is still conditional.  The bare words "Stage IV" must not turn that
        # sentence into affirmative metastatic evidence.
        if re.search(
            r"\b(?:biopsy|work[\s-]?up|staging|imaging)\b[^.;]{0,90}"
            r"\b(?:for|to)\b[^.;]{0,20}\b(?:confirm\w*|establish\w*|definitive)\b"
            r"[^.;]{0,40}\bstage\s*(?:iv|4)\b|"
            r"\bstage\s*(?:iv|4)\b[^.;]{0,70}"
            r"\b(?:pending|awaiting|confirmation|confirmatory|biopsy)\b",
            context,
        ):
            continue
        return True
    return False


def align_stage_with_confirmed_distant(stage, distant, cancer_type):
    """Make a non-IV stage longitudinally consistent with confirmed current M1 disease."""
    stage = str(stage or "").strip()
    if not is_confirmed_distant_value(distant, cancer_type):
        return stage, None
    if re.search(r"\bstage\s*(?:iv|4)\b|\bmetastatic\b", stage, re.IGNORECASE):
        return stage, None
    if not stage or stage.lower() in {
        "not staged", "not staged in note", "not mentioned", "not available",
        "not specified", "unknown",
    }:
        return "Stage IV (metastatic)", "confirmed distant disease fills Stage IV"
    return (
        f"Originally {stage}; now Stage IV (metastatic)",
        "confirmed distant disease updates historical/nonmetastatic stage",
    )


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


def _general_mentions_regional_nodes(value, cancer_type):
    """Return whether the broad Metastasis value makes a regional-node claim."""
    value = str(value or "").lower()
    value = re.sub(
        r"(?:contralateral(?:\s+(?:right|left))?\s+(?:axill\w*|supraclavicular)|"
        r"cervical|mediastinal)(?:\s+(?:lymph\s*)?nodes?)?",
        "",
        value,
    )
    if cancer_type == "breast":
        site_pattern = (
            r"regional|ipsilateral|axill\w*|sentinel|subpectoral|supraclavicular|"
            r"infraclavicular|internal mammary"
        )
    else:
        site_pattern = (
            r"regional|peripancreatic|periportal|porta hepatis|portacaval|upper abdominal|"
            r"mesenteric|perigastric|"
            r"common hepatic|celiac"
        )
    return bool(
        re.search(rf"(?:{site_pattern})[^.;]{{0,35}}(?:lymph[\s-]*node|\bnodal\b|\bnodes?\b)", value)
        or re.search(r"(?:lymph[\s-]*node|\bnodal\b)[^.;]{0,35}\bregional\b", value)
        or re.search(r"\b(?:confirmed|suspected|involved)\s+regional\s+(?:lymph[\s-]*)?nodes?\b", value)
    )


def _regional_node_context_allowed(context, cancer_type):
    """Reject negative, nonregional, or clearly other-primary node evidence."""
    context = str(context or "").lower()
    if re.search(
        r"\b(?:0\s*(?:/|of)\s*\d+|n0|nx)\b|negative for (?:carcinoma|malignancy|metastasis)|"
        r"\b(?:benign|reactive|granulomatous|sarcoid)\b|"
        r"no (?:evidence of )?(?:regional )?(?:nodal|lymph[\s-]*node) (?:disease|involvement|metastasis)",
        context,
    ):
        return False
    current_terms = (("breast", "mammary") if cancer_type == "breast"
                     else ("pancrea", "pdac", "whipple", "pancreaticoduoden"))
    other_terms = _other_primary_terms(cancer_type)
    if any(term in context for term in other_terms):
        return False
    if cancer_type == "breast":
        if re.search(r"\bcontralateral\b[^.;]{0,30}(?:axill\w*|supraclavicular)", context):
            return False
        return bool(re.search(
            r"\b(?:regional|ipsilateral|axill\w*|sentinel|subpectoral|supraclavicular|"
            r"infraclavicular|internal mammary|slns?)\b",
            context,
        ))
    if re.search(r"\b(?:cervical|mediastinal|supraclavicular)\b[^.;]{0,25}(?:lymph[\s-]*node|nodes?|nodal)", context):
        return False
    return bool(
        re.search(
            r"\b(?:regional|peripancreatic|periportal|porta hepatis|portacaval|upper abdominal|"
            r"mesenteric|perigastric|"
            r"common hepatic|celiac)\b",
            context,
        )
        or (
            re.search(r"\b(?:lymph[\s-]*nodes?|nodal)\b", context)
            and any(term in context for term in current_terms)
        )
    )


def _historical_node_context(context):
    return bool(re.search(
        r"\b(?:history of|historical|previously|prior|originally|at diagnosis|status post|s/p|"
        r"underwent|resected|resection|surgical pathology|in 20\d{2})\b|\b20\d{2}[-/]\d{1,2}",
        str(context or "").lower(),
    ))


def _regional_evidence_context(text, start, end, lookback=220):
    """Include the immediately preceding imaging/pathology context for terse result sentences."""
    sentence_start, sentence_end = _sentence_bounds(text, start, end)
    return text[max(0, sentence_start - lookback):sentence_end]


def final_regional_node_evidence(cancer_type, stage, note_text, assessment_and_plan):
    """Classify regional-node evidence for the final broad-Metastasis sanitizer.

    Pathologic TN/count/biopsy evidence is distinguished from explicit radiographic nodal
    metastases and from cN/imaging suspicion. Full-note pathology may be historical; current
    A/P evidence is considered first.
    """
    stage = str(stage or "")
    path_tn = re.search(
        r"\b((?:yp|p)T\d[a-d]?(?:\([^)]*\))?\s*,?\s*(?:yp|p)?N([1-3])([a-c]?))\b",
        stage,
        re.IGNORECASE,
    )
    standalone_pn = re.search(r"\b((?:yp|p)N([1-3])([a-c]?))\b", stage, re.IGNORECASE)
    if path_tn or standalone_pn:
        match = path_tn or standalone_pn
        detail_match = re.search(r"(?:yp|p)?N[1-3][a-c]?", match.group(1), re.IGNORECASE)
        return "HISTORICAL_PATHOLOGIC", detail_match.group(0) if detail_match else match.group(1)

    count_patterns = (
        re.compile(
            r"\b([1-9]\d*)\s*(?:/|of)\s*(\d+)\s*(?:regional\s+)?(?:lymph[\s-]*)?"
            r"(?:nodes?|lns?|slns?)?\s*(?:were\s+)?(?:positive|involved|metastatic|"
            r"with\s+(?:micro|macro)?metasta\w*)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([1-9]\d*)\s*(?:/|of)\s*(\d+)\s*(?:positive|involved|metastatic)\s+"
            r"(?:regional\s+)?(?:lymph[\s-]*)?(?:nodes?|lns?|slns?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([1-9]\d*)\s*(?:/|of)\s*(\d+)\s*(?:slns?|lns?|sentinel\s+(?:lymph\s*)?nodes?|"
            r"lymph[\s-]*nodes?)\s*\+\s*(?:\([^)]*(?:micro|macro)?metasta\w*[^)]*\))?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([1-9]\d*)\s*(?:/|of)\s*(\d+)\s*(?:sentinel\s+)?(?:lymph[\s-]*)?nodes?\s+"
            r"(?:were\s+)?with\s+(?:micro|macro)?metasta\w*",
            re.IGNORECASE,
        ),
    )
    biopsy_malignancy = re.compile(
        r"(?:fna|fine needle aspiration|core biopsy|biopsy)[^.;\n]{0,120}"
        r"(?:lymph[\s-]*node|\bln\b|\bnodal\b|\bnodes?\b)[^.;\n]{0,100}"
        r"(?:metastatic|positive for|involved by|malignan\w*|adenocarcinoma|carcinoma)|"
        r"(?:lymph[\s-]*node|\bln\b|\bnodal\b|\bnodes?\b)[^.;\n]{0,100}"
        r"(?:fna|fine needle aspiration|core biopsy|biopsy|positive for|involved by)[^.;\n]{0,100}"
        r"(?:metastatic|malignan\w*|adenocarcinoma|carcinoma)",
        re.IGNORECASE,
    )
    note_pathologic_n = re.compile(
        r"\b((?:(?:yp|p)T\d[a-d]?(?:\([^)]*\))?\s*,?\s*(?:yp|p)?N[1-3][a-c]?)|"
        r"(?:(?:yp|p)N[1-3][a-c]?))\b",
        re.IGNORECASE,
    )

    if cancer_type == "breast":
        micro_positive_node = re.compile(
            r"\bpositive\s+(?:lns?|lymph[\s-]*nodes?|nodes?)\b[^.;\n]{0,80}"
            r"\bmicrometasta\w*\b|"
            r"\bmicrometasta\w*\b[^.;\n]{0,80}\bpositive\s+"
            r"(?:lns?|lymph[\s-]*nodes?|nodes?)\b",
            re.IGNORECASE,
        )
        ap_text = str(assessment_and_plan or "")
        for match in micro_positive_node.finditer(ap_text):
            start, end = _sentence_bounds(ap_text, match.start(), match.end())
            context = ap_text[start:end]
            pre = ap_text[max(start, match.start() - 50):match.start()]
            if any(term in context.lower() for term in _other_primary_terms(cancer_type)):
                continue
            if re.search(r"\b(?:if|whether|patients?\s+with)\b", pre, re.IGNORECASE):
                continue
            return "HISTORICAL_PATHOLOGIC", "micrometastatic regional node"

    for source_name, source in (("ap", assessment_and_plan), ("note", note_text)):
        text = str(source or "")
        for match in note_pathologic_n.finditer(text):
            start, end = _sentence_bounds(text, match.start(), match.end())
            context = text[start:end].lower()
            if any(term in context for term in _other_primary_terms(cancer_type)):
                continue
            detail_match = re.search(r"(?:yp|p)?N[1-3][a-c]?", match.group(1), re.IGNORECASE)
            return "HISTORICAL_PATHOLOGIC", detail_match.group(0) if detail_match else match.group(1)
        for pattern in count_patterns:
            for match in pattern.finditer(text):
                context = _regional_evidence_context(text, match.start(), match.end())
                if not _regional_node_context_allowed(context, cancer_type):
                    continue
                evidence = "HISTORICAL_PATHOLOGIC" if _historical_node_context(context) else "PATHOLOGIC"
                if source_name == "note" and not str(assessment_and_plan or "").strip():
                    evidence = "HISTORICAL_PATHOLOGIC"
                detail = f"{match.group(1)}/{match.group(2)} nodes positive"
                if re.search(r"micrometasta", match.group(0), re.IGNORECASE):
                    detail += " (micrometastasis)"
                return evidence, detail
        for match in biopsy_malignancy.finditer(text):
            start, end = _sentence_bounds(text, match.start(), match.end())
            context = text[start:end]
            if not _regional_node_context_allowed(context, cancer_type):
                continue
            evidence = "HISTORICAL_PATHOLOGIC" if _historical_node_context(context) else "PATHOLOGIC"
            site = re.search(
                r"(?:regional|ipsilateral|axill\w*|sentinel|subpectoral|supraclavicular|"
                r"infraclavicular|internal mammary|slns?|peripancreatic|periportal|porta hepatis|portacaval|"
                r"upper abdominal|"
                r"mesenteric|perigastric)[^.;\n]{0,25}(?:lymph[\s-]*node|nodes?|nodal)",
                context,
                re.IGNORECASE,
            )
            return evidence, site.group(0) if site else "positive regional node biopsy"

    clinical_n = re.search(r"\bcN([1-3])([a-c]?)\b", stage, re.IGNORECASE)
    texts = (str(assessment_and_plan or ""), str(note_text or ""))
    explicit_nodal_metastasis = re.compile(
        r"\b(?:nodal|lymph[\s-]*node) metastas\w*\b|"
        r"\bmetastatic\s+(?:regional\s+)?(?:lymph[\s-]*)?nodes?\b|"
        r"\b(?:lymph[\s-]*nodes?|nodal disease)\b[^.;\n]{0,55}"
        r"(?:consistent with|involved by|positive for)\s+(?:nodal\s+)?metastas\w*",
        re.IGNORECASE,
    )
    suspected_nodes = re.compile(
        r"\b(?:suspicious|concerning|indeterminate|possible|equivocal)\b[^.;\n]{0,60}"
        r"(?:lymph[\s-]*nodes?|nodal disease)|"
        r"(?:lymph[\s-]*nodes?|nodal disease)[^.;\n]{0,60}"
        r"\b(?:suspicious|concerning|indeterminate|possible|equivocal)\b",
        re.IGNORECASE,
    )
    imaging_anchor = re.compile(r"\b(?:ct|mri|pet(?:/ct)?|scan|imaging|radiograph\w*)\b", re.IGNORECASE)
    for text in texts:
        for match in explicit_nodal_metastasis.finditer(text):
            context = _regional_evidence_context(text, match.start(), match.end(), lookback=320)
            if not _regional_node_context_allowed(context, cancer_type):
                continue
            if UNCERTAINTY_RE.search(context):
                return "SUSPECTED", "imaging-suspicious regional nodes"
            evidence = "RADIOGRAPHIC" if imaging_anchor.search(context) else "DOCUMENTED_MALIGNANT"
            return evidence, "regional nodal metastases"
        for match in suspected_nodes.finditer(text):
            start, end = _sentence_bounds(text, match.start(), match.end())
            context = text[start:end]
            if _regional_node_context_allowed(context, cancer_type):
                return "SUSPECTED", "imaging-suspicious regional nodes"
    if clinical_n:
        return "SUSPECTED", clinical_n.group(0)
    return None, ""


def _supported_locoregional_clause(general, note_text, assessment_and_plan, cancer_type):
    """Preserve a true breast locoregional/chest-wall recurrence through final rebuilding."""
    if cancer_type != "breast":
        return ""
    general = str(general or "")
    requested = re.search(
        r"loc(?:al[\s-]*)?regional(?:\s+chest[\s-]*wall)? recurrence|"
        r"chest[\s-]*wall (?:recurrence|lesion)|parasternal",
        general,
        re.IGNORECASE,
    )
    ap_supported = re.search(
        r"loc(?:al[\s-]*)?regional(?:\s+chest[\s-]*wall)? recurrence|"
        r"chest[\s-]*wall recurrence",
        str(assessment_and_plan or ""),
        re.IGNORECASE,
    )
    if ap_supported:
        return "locoregional chest-wall recurrence" if "chest" in ap_supported.group(0).lower() \
            else "locoregional recurrence"
    if not requested:
        return ""
    supported = re.search(
        r"loc(?:al[\s-]*)?regional(?:\s+chest[\s-]*wall)? recurrence|"
        r"chest[\s-]*wall recurrence",
        str(note_text or ""),
        re.IGNORECASE,
    )
    if not supported:
        return ""
    return "locoregional chest-wall recurrence" if "chest" in supported.group(0).lower() \
        else "locoregional recurrence"


_DM_SITE_PATTERNS = {
    "cervical_nodes": r"\b(?:(?:right|left|bilateral)\s+)?(?:cervical|level\s+v\s*b)\s+(?:lymph\s*)?nodes?\b",
    "contralateral_nodes": r"\bcontralateral[^.;,]{0,30}(?:axill\w*|supraclavicular)(?:\s+(?:lymph\s*)?nodes?)?\b",
    "regional_nodes": r"\b(?:(?:right|left|ipsilateral)\s+)?(?:axill\w*|supraclavicular|sentinel|subpectoral|infraclavicular|internal mammary|regional)(?:\s+(?:lymph\s*)?nodes?)?\b",
    "peritoneum": r"\b(?:peritoneum|peritoneal)(?:\s+(?:disease|implants?|carcinomatosis|metasta\w*))?\b",
    "omentum": r"\b(?:omentum|omental)(?:\s+(?:disease|implants?|caking|metasta\w*))?\b",
    "liver": r"\b(?:liver|hepatic)(?!\s+(?:artery|vein|duct|function))(?:\s+(?:lesions?|metasta\w*))?\b",
    "lung": r"\b(?:lungs?|pulmonary)(?!\s+(?:artery|embol\w*|function))(?:\s+(?:nodules?|lesions?|metasta\w*))?\b",
    "bone": r"\b(?:bone|osseous|ilium|iliac|sacrum|sacral|spine|vertebr\w*|ribs?|sternum|femur)(?:\s+(?:lesions?|metasta\w*|disease|ala))?\b",
    "brain": r"\b(?:brain|cerebral|intracranial|falx(?:\s+cerebri)?|falcine|parafalcine|dural(?:-based)?)\b",
    "pleura": r"\bpleur\w*(?:\s+(?:disease|metasta\w*))?\b",
    "adrenal": r"\badrenal(?:\s+(?:glands?|nodules?|lesions?|metasta\w*))?\b",
    "abdominal_wall": r"\babdominal[ -]*wall(?:\s+(?:lesion|metasta\w*))?\b",
    "chest_wall": r"\b(?:parasternal|chest[ -]*wall)(?:\s+(?:lesion|nodule|recurrence|metasta\w*))?\b",
}
_DM_UNCERTAIN_RE = re.compile(
    r"\b(?:not sure|unsure|suspect\w*|suspicious|possible|uncertain|equivocal|pending|"
    r"cannot exclude|concern\w*|indeterminate|suggestive|likely|presumptiv\w*|"
    r"differential|too small|hard to interpret|non[- ]avid|not pet[- ]avid)\b|"
    r"\bconsistent with\b", re.IGNORECASE,
)
_DM_HISTORY_RE = re.compile(
    r"\b(?:history of|historical|previously|prior|originally|at diagnosis|status post|s/p)\b",
    re.IGNORECASE,
)
_DM_NO_RE = re.compile(
    r"(?:^|[^A-Za-z])(?:c|p|yp)?M0\b|\bno (?:other )?(?:sites? of )?(?:distant |systemic )?"
    r"(?:metasta\w*|disease)\b|\bno evidence of (?:distant |systemic )?"
    r"(?:metasta\w*|disease)\b|\bwithout (?:evidence of )?(?:distant |systemic )?"
    r"(?:metasta\w*|disease)\b", re.IGNORECASE,
)
def _dm_mentions(value):
    value = str(value or "")
    found, occupied = [], []
    for key, pattern in _DM_SITE_PATTERNS.items():
        for match in re.finditer(pattern, value, re.IGNORECASE):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            occupied.append(match.span())
            found.append((match.start(), match.end(), key, match.group(0)))
    return sorted(found)
def _dm_claims(value):
    value, claims, seen = str(value or "").strip(), [], set()
    field_status = met_status(value)
    labels = {
        "peritoneum": "peritoneum", "omentum": "omentum", "liver": "liver",
        "lung": "lung", "bone": "bone", "pleura": "pleura",
        "adrenal": "adrenal gland", "abdominal_wall": "abdominal wall",
        "chest_wall": "parasternal/chest-wall disease",
        "regional_nodes": "regional lymph nodes",
    }
    for start, end, key, raw in _dm_mentions(value):
        if key in seen:
            continue
        seen.add(key)
        s, e = _sentence_bounds(value, start, end)
        uncertain = field_status == "UNSURE" or bool(_DM_UNCERTAIN_RE.search(value[s:e]))
        label = labels.get(key, raw.lower())
        if key == "brain" and re.search(r"falx|falcine|parafalcine|dural", raw, re.I):
            label = "falx/dural lesion"
        claims.append({"key": key, "label": label,
                       "certainty": "SUSPECTED" if uncertain else "CONFIRMED"})
    return claims
def _dm_contexts(text, key, cancer_type):
    text, contexts = str(text or ""), []
    for match in re.finditer(_DM_SITE_PATTERNS[key], text, re.IGNORECASE):
        start, end = _sentence_bounds(text, match.start(), match.end())
        context = text[start:end].strip()
        other = any(term in context.lower() for term in _other_primary_terms(cancer_type))
        target = ("breast", "mammary", "ductal", "lobular") if cancer_type == "breast" \
            else ("pancrea", "pdac", "whipple")
        if context and not (other and not any(term in context.lower() for term in target)):
            contexts.append(context)
    if key == "liver":
        for match in re.finditer(r"\bliver\s*:", text, re.IGNORECASE):
            contexts.append(text[match.start():min(len(text), match.end() + 650)])
    return list(dict.fromkeys(contexts))
def _dm_flags(contexts, key):
    texts = [context.lower() for context in contexts]
    text = "\n".join(texts)
    path = any(
        re.search(r"\b(?:biopsy|fna|cytolog\w*|patholog\w*)\b", context)
        and re.search(r"\b(?:positive for|confirmed|metastatic|malignan\w*|adenocarcinoma|carcinoma)\b", context)
        and not re.search(r"\b(?:negative for|no evidence of|benign|reactive)\b", context)
        for context in texts
    )
    uncertain = any(_DM_UNCERTAIN_RE.search(context) for context in texts)
    benign = bool(re.search(r"\b(?:benign|reactive|granulomatous|sarcoid)\b", text))
    if key == "liver":
        benign |= bool(re.search(r"\b(?:hemangioma|simple cyst|consistent with (?:a )?cyst)\b", text))
    elif key == "brain":
        benign |= bool(re.search(r"\b(?:meningioma|encephalomalacia|gliosis)\b", text))
    negative = bool(
        re.search(r"\bno (?:new )?(?:suspicious )?(?:lesions?|nodules?|metastases|abnormalities)\b", text)
        or re.search(r"\bno evidence of (?:malignan\w*|metasta\w*)\b", text)
        or re.search(r"\bnegative for (?:malignan\w*|carcinoma|metasta\w*)\b", text)
    )
    special = {
        "peritoneum": r"peritoneal carcinomatosis|carcinomatosis with peritoneal|peritoneal implants?",
        "omentum": r"omental cak(?:e|ing)|carcinomatosis with [^.;]{0,30}omental|omental implants?",
        "lung": r"lung predominant disease|pulmonary nodules consistent with treated metastases",
    }
    site = _DM_SITE_PATTERNS[key]
    definite = any(
        not _DM_UNCERTAIN_RE.search(context) and (
            (key in special and re.search(special[key], context))
            or re.search(rf"(?:known|treated|confirmed|biopsy[- ]proven)[^.;]{{0,45}}{site}[^.;]{{0,35}}metasta\w*", context)
            or re.search(rf"{site}[^.;]{{0,45}}(?:metastases|metastatic disease|carcinomatosis)", context)
            or re.search(rf"metasta\w*[^.;]{{0,40}}(?:to|in|involving)\s+(?:the\s+)?{site}", context)
        ) for context in texts
    )
    direct = bool(
        re.search(r"\b(?:invad\w*|extension|abut\w*|encas\w*|inseparable from)\b", text)
        and not re.search(r"\bmetasta\w*\b", text)
    )
    return path, uncertain, definite, benign, negative, bool(_DM_HISTORY_RE.search(text)), direct
def _dm_current_no(note_text, assessment_and_plan):
    for text in (str(assessment_and_plan or ""), str(note_text or "")):
        for match in _DM_NO_RE.finditer(text):
            start, end = _sentence_bounds(text, match.start(), match.end())
            if not _DM_HISTORY_RE.search(text[start:end]):
                return True
    return False
def _dm_generic_m1(assessment_and_plan, cancer_type):
    text = str(assessment_and_plan or "")
    if has_explicit_m1_evidence(text, cancer_type):
        return True
    pattern = r"\bmetastatic\s+(?:adenocarcinoma|carcinoma)(?:\s+of\s+the\s+(?:pancreas|breast))?\b"
    return any(
        not _DM_UNCERTAIN_RE.search(text[s:e]) and not _DM_HISTORY_RE.search(text[s:e])
        for match in re.finditer(pattern, text, re.IGNORECASE)
        for s, e in [_sentence_bounds(text, match.start(), match.end())]
    )

def _dm_site_evidence(key, note_text, assessment_and_plan, cancer_type):
    if cancer_type == "breast" and key in ("regional_nodes", "chest_wall"):
        return "REMOVE"
    ap = _dm_contexts(assessment_and_plan, key, cancer_type)
    note = _dm_contexts(note_text, key, cancer_type)
    if not ap and not note:
        return "REMOVE"
    af, nf = _dm_flags(ap, key), _dm_flags(note, key)
    ap_path, ap_uncertain, ap_definite, ap_benign, ap_negative, _, ap_direct = af
    no_path, no_uncertain, no_definite, no_benign, no_negative, no_history, no_direct = nf
    current_no = _dm_current_no(note_text, assessment_and_plan)
    if ap_path:
        return "CONFIRMED"
    if ap_uncertain:
        return "CONFIRMED" if key in ("peritoneum", "omentum") and no_definite else "SUSPECTED"
    if ap_definite:
        return "CONFIRMED"
    if ap_benign or ap_negative or ap_direct:
        return "HISTORICAL" if no_path else "REMOVE"
    if no_path:
        return "HISTORICAL" if no_history else "CONFIRMED"
    if no_definite:
        return "CONFIRMED"
    if no_uncertain:
        if (
            current_no
            and not _dm_generic_m1(assessment_and_plan, cancer_type)
            and (no_benign or no_negative)
        ):
            return "REMOVE"
        return "SUSPECTED"
    if no_benign or no_negative or no_direct or current_no:
        return "REMOVE"
    return "KEEP"

def _dm_render(states):
    groups = {
        status: list(dict.fromkeys(s["label"] for s in states if s["status"] == status))
        for status in ("CONFIRMED", "HISTORICAL", "SUSPECTED")
    }
    join = lambda xs: xs[0] if len(xs) == 1 else f"{', '.join(xs[:-1])} and {xs[-1]}"
    clauses = []
    if groups["CONFIRMED"]:
        clauses.append(f"Yes, to {join(groups['CONFIRMED'])}")
    if groups["HISTORICAL"]:
        clauses.append(f"{'Yes, ' if not clauses else ''}historically confirmed to {join(groups['HISTORICAL'])}")
    if groups["SUSPECTED"]:
        clauses.append(f"Not sure/Suspected, to {join(groups['SUSPECTED'])}")
    return "; ".join(clauses)

def sanitize_distant_metastasis_by_site(
    distant, general, stage, note_text, assessment_and_plan, cancer_type
):
    _ = stage
    distant, general = str(distant or "").strip(), str(general or "").strip()
    reasons, changed = [], False
    if cancer_type == "breast" and re.search(
        r"carotid\s+(?:body\s+)?(?:tumou?r|paraganglioma)|"
        r"carotid\s+bifurcation\s+mass[^.;]{0,100}(?:longstanding|stable|paraganglioma)",
        f"{assessment_and_plan or ''}\n{note_text or ''}",
        re.IGNORECASE,
    ):
        cleaned_distant = re.sub(
            r"(?i)(?:,\s*|\band\s+)?(?:left\s+|right\s+)?carotid\s+artery\s+bifurcation",
            "",
            distant,
        )
        cleaned_distant = re.sub(r"\s+,", ",", cleaned_distant)
        cleaned_distant = re.sub(r",\s*,+", ",", cleaned_distant)
        cleaned_distant = re.sub(r"\s{2,}", " ", cleaned_distant).strip(" ,;")
        if cleaned_distant != distant:
            distant = cleaned_distant
            changed = True
            reasons.append("removed carotid-body/paraganglioma alternative diagnosis")
    field_status, claims = met_status(distant), _dm_claims(distant)
    had_distant_sites = bool(claims)
    by_key = {claim["key"]: claim for claim in claims}
    if field_status not in ("NO", "EMPTY"):
        ceiling = "CONFIRMED" if field_status == "YES" else "SUSPECTED"
        for claim in _dm_claims(general):
            if claim["key"] in by_key:
                continue
            if ceiling == "SUSPECTED":
                claim["certainty"] = "SUSPECTED"
            claim["from_general"] = True
            claims.append(claim)
            by_key[claim["key"]] = claim

    states = []
    for claim in claims:
        evidence = _dm_site_evidence(
            claim["key"], note_text, assessment_and_plan, cancer_type
        )
        from_general = claim.get("from_general", False)
        if evidence == "REMOVE":
            if not from_general:
                changed = True
                reasons.append(f"removed unsupported/negative {claim['label']}")
            continue
        if evidence == "KEEP":
            if from_general:
                continue
            status = claim["certainty"]
        elif evidence == "SUSPECTED":
            status = "SUSPECTED"
            if claim["certainty"] == "CONFIRMED":
                changed = True
                reasons.append(f"downgraded {claim['label']} to suspected")
        elif evidence == "HISTORICAL":
            status = "HISTORICAL" if claim["certainty"] == "CONFIRMED" else "SUSPECTED"
            changed |= status != claim["certainty"]
        else:
            status = claim["certainty"]
        if from_general:
            changed = True
            reasons.append(f"preserved existing general-field site {claim['label']}")
        states.append({"label": claim["label"], "status": status})

    if states:
        return (_dm_render(states), reasons) if changed else (distant, [])
    if not had_distant_sites:
        return distant, []
    if field_status == "YES" and _dm_generic_m1(assessment_and_plan, cancer_type):
        fallback = "Yes"
    elif _dm_current_no(note_text, assessment_and_plan):
        fallback = "No"
    else:
        fallback = "Not sure"
    if fallback != distant:
        reasons.append("recalibrated site-free distant status")
    return fallback, reasons


def sanitize_general_metastasis(
    distant, general, stage, note_text, assessment_and_plan, cancer_type
):
    """Rebuild broad Metastasis from supported regional claims and the Distant ceiling."""
    distant = str(distant or "").strip()
    general = str(general or "").strip()
    reasons = []
    clauses = []

    locoregional = _supported_locoregional_clause(
        general, note_text, assessment_and_plan, cancer_type
    )
    if locoregional:
        clauses.append(locoregional)

    regional_requested = _general_mentions_regional_nodes(general, cancer_type)
    evidence, detail = final_regional_node_evidence(
        cancer_type, stage, note_text, assessment_and_plan
    )
    strong_regional_evidence = evidence in (
        "HISTORICAL_PATHOLOGIC", "PATHOLOGIC", "RADIOGRAPHIC", "DOCUMENTED_MALIGNANT"
    )
    if regional_requested or strong_regional_evidence:
        suffix = f" ({detail})" if detail else ""
        if evidence == "HISTORICAL_PATHOLOGIC":
            clauses.append(f"historical pathologically confirmed regional lymph-node involvement{suffix}")
        elif evidence == "PATHOLOGIC":
            clauses.append(f"pathologically confirmed regional lymph-node involvement{suffix}")
        elif evidence == "RADIOGRAPHIC":
            clauses.append("radiographically involved regional lymph nodes")
        elif evidence == "DOCUMENTED_MALIGNANT":
            clauses.append("documented malignant regional lymph-node involvement")
        elif evidence == "SUSPECTED":
            clauses.append(f"clinically suspected regional lymph-node involvement{suffix}")
            reasons.append("downgraded unsupported confirmed regional claim")
        else:
            reasons.append("removed unsupported regional claim")

    if is_confirmed_distant_value(distant, cancer_type):
        rebuilt = "; ".join([distant] + clauses) if clauses else distant
    else:
        distant_status = met_status(distant)
        if distant_status in ("UNSURE", "MIXED"):
            rebuilt = (
                f"Yes, {'; '.join(clauses)}; distant disease uncertain — {distant}"
                if clauses else distant
            )
        elif distant_status == "NO":
            rebuilt = f"Yes, {'; '.join(clauses)}; no distant metastasis" if clauses else "No"
        elif distant_status == "EMPTY":
            rebuilt = f"Yes, {'; '.join(clauses)}" if clauses else "Not sure"
        else:
            rebuilt = f"Yes, {'; '.join(clauses)}" if clauses else distant

    if rebuilt != general and not reasons:
        reasons.append("rebuilt from regional evidence and Distant ceiling")
    return rebuilt, reasons


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


UNTREATED_RESPONSE_RE = re.compile(
    r"\bnot yet on treatment\b|\bnot on treatment\b|\bno response to assess\b",
    re.IGNORECASE,
)

TUMOR_RESPONSE_ANCHOR_RE = re.compile(
    r"\b(?:cancer|disease|tumou?r|mass|lesion|metasta\w*|recurr\w*|carcinoma|"
    r"lymph[\s-]*node|nodule|ca\s*19[\s-]*9|cea|respond\w*|response|progress\w*|"
    r"stable disease|progressive disease|partial response|complete response)\b",
    re.IGNORECASE,
)

NON_RESPONSE_FINDING_RE = re.compile(
    r"hand[\s-]*foot|mucositis|neuropath|nausea|vomit|diarrhea|fatigue|toxicit|"
    r"side effect|pneumobilia|biliary duct|bile duct|cholangit|jaundice|"
    r"post[\s-]*(?:operative|surgical)|surgical recovery|healing|seroma|hematoma|"
    r"free fluid|cut edge of (?:the )?pancreas",
    re.IGNORECASE,
)


def _clinical_sentences(text):
    """Split prose conservatively while preserving measurement-rich source wording."""
    return [
        sentence.strip(" \t-*\u2022")
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+|\s{2,}-\s+", str(text or ""))
        if sentence.strip(" \t-*\u2022")
    ]


def _first_supported_sentence(sources, predicate):
    for source in sources:
        for sentence in _clinical_sentences(source):
            if predicate(sentence):
                return sentence if sentence.endswith((".", "!", "?")) else sentence + "."
    return ""


def _positive_recurrence_or_growth_sentence(assessment_and_plan, findings, note_text):
    """Find a current, affirmative recurrence/growth statement, excluding risk and uncertainty."""
    excluded = re.compile(
        r"no evidence of (?:disease )?recurrence|without recurrence|free of recurrence|"
        r"risk (?:of|for) recurrence|high risk for recurrence|concern\w* for recurrence|"
        r"possible recurrence|suspect\w* recurrence|if (?:the )?(?:cancer|disease) recur|"
        r"whether (?:the )?(?:cancer|disease) (?:has )?recur|"
        r"does not recur|do not recur|prevent\w* recurrence|decreas\w* the risk of recurrence",
        re.IGNORECASE,
    )
    recurrence = re.compile(
        r"local[\s-]*regional recurrence|locoregional recurrence|local recurrence|"
        r"recurrent (?:cancer|disease|carcinoma)|compatible with recurrent disease|"
        r"biopsy[^.;]{0,80}\brecurr\w*",
        re.IGNORECASE,
    )
    growth = re.compile(
        r"(?:growth|grew|enlarged|interval increase|increas\w* (?:in )?size)[^.;]{0,100}"
        r"(?:cancer|tumou?r|mass|lesion)|"
        r"(?:cancer|tumou?r|mass|lesion)[^.;]{0,100}"
        r"(?:growth|grew|enlarged|interval increase|increas\w* (?:in )?size)",
        re.IGNORECASE,
    )
    current_source = "\n".join(
        str(value or "") for value in (assessment_and_plan, findings)
    )
    all_source = f"{current_source}\n{note_text or ''}"
    source_sentences = _clinical_sentences(current_source)
    confirmed_recurrence = any(
        recurrence.search(sentence) and not excluded.search(sentence)
        for sentence in source_sentences
    )
    prior_treatment = bool(
        re.search(
            r"\b(?:progressed|progression) on\b|\brecurr\w* after\b|"
            r"\b(?:completed|finished|received)\b[^.;\n]{0,60}"
            r"(?:chemo|radiation|therapy|treatment|cycle)|"
            r"\b(?:previously|prior) (?:treated|therapy|treatment|chemo|radiation)|"
            r"\bs/p\s+[^.;\n]{0,45}(?:chemo|radiation|therapy|treatment)",
            all_source,
            re.IGNORECASE,
        )
    )

    def supported(sentence):
        return not excluded.search(sentence) and bool(
            recurrence.search(sentence)
            or ((prior_treatment or confirmed_recurrence) and growth.search(sentence))
        )

    return _first_supported_sentence((assessment_and_plan, findings), supported)


def _same_day_treatment_started(assessment_and_plan, recent_changes, current_meds):
    """Recognize actual same-day treatment starts, not prescriptions or future plans."""
    context = f"{recent_changes or ''}\n{assessment_and_plan or ''}"
    current_meds = str(current_meds or "").strip()
    if not current_meds:
        return False
    med_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", current_meds)
        if token.lower() not in {"systemic", "therapy", "currently", "hold", "on"}
    ]
    for sentence in _clinical_sentences(context):
        explicit_administration = re.search(
            r"\b(?:received|administered|was given|given|began|initiated|started)\b"
            r"[^.;\n]{0,80}\b(?:today|this visit)\b|"
            r"\b(?:today|this visit)\b[^.;\n]{0,50}\b(?:received|administered|was given)\b|"
            r"^start\s+[^.;\n]{1,80}\btoday\b",
            sentence,
            re.IGNORECASE,
        )
        future_only = re.search(
            r"\b(?:will|plan(?:ned)? to|recommend(?:ed)?|consider|may|can|prescri\w*|order\w*)\b"
            r"[^.;\n]{0,20}\b(?:start|begin|initiat)",
            sentence,
            re.IGNORECASE,
        )
        treatment_named = bool(
            re.search(
                r"\b(?:chemo(?:therapy)?|treatment|infusion|injection|cycle\s*\d+)\b",
                sentence,
                re.I,
            )
            or any(
                re.search(rf"(?<!\w){re.escape(token)}(?!\w)", sentence, re.IGNORECASE)
                for token in med_tokens
            )
        )
        if explicit_administration and not future_only and treatment_named:
            return True
    return False


def _explicit_treatment_linked_mass_disappearance(assessment_and_plan, findings, note_text):
    def supported(sentence):
        lower = sentence.lower()
        return bool(
            re.search(r"\b(?:mass|tumou?r|lesion)\b", lower)
            and re.search(r"\bno longer (?:well )?(?:seen|visible)\b", lower)
            and re.search(r"\b(?:likely|consistent(?:ly)?) (?:related|due) to treatment\b", lower)
        )

    return _first_supported_sentence((assessment_and_plan, findings), supported)


def _direct_tumor_status_sentence(assessment_and_plan, findings, note_text):
    direct_status = re.compile(
        r"\bno evidence of (?:metastatic disease|disease|recurrence)\b|"
        r"\b(?:stable|progressive) disease\b",
        re.IGNORECASE,
    )
    return _first_supported_sentence(
        (assessment_and_plan, findings),
        lambda sentence: bool(direct_status.search(sentence)),
    )


def _has_affirmative_progression(response):
    for sentence in _clinical_sentences(response):
        if not re.search(
            r"\bprogress(?:ion|ed|ing)\b|\bmixed response\b|"
            r"\bnew (?:[a-z][\w/-]*\s+){0,3}(?:metasta\w*|tumou?r|mass|lesion)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        if re.search(
            r"\bno (?:evidence of )?progression\b|\bwithout progression\b|"
            r"\bnot progressing\b|"
            r"\bno new (?:[a-z][\w/-]*\s+){0,3}(?:metasta\w*|tumou?r|mass|lesion)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        return True
    return False


def clear_held_anticancer_meds(current_meds, note_text, assessment_and_plan):
    """Clear a chemotherapy regimen when the source explicitly says it is held/paused."""
    current_meds = str(current_meds or "").strip()
    if not current_meds:
        return current_meds, None
    source = f"{note_text or ''}\n{assessment_and_plan or ''}".lower()
    held = re.search(
        r"pause\s+(?:the\s+)?systemic\s+therapy|"
        r"hold\w*\s+(?:her|his|the)?\s*chemo(?:therapy)?\b|"
        r"holding\s+(?:her|his)?\s*chemo|"
        r"systemic\s+therapy\s+(?:is\s+)?(?:currently\s+)?(?:on\s+)?hold|"
        r"chemotherapy\s+(?:is\s+)?(?:currently\s+)?(?:on\s+)?hold",
        source,
    )
    active_now = re.search(
        r"presents?\s+for\s+c\d|today'?s?\s+(?:infusion|cycle)|"
        r"proceed with\s+(?:today'?s?\s+)?(?:chemo|treatment|cycle)|"
        r"continue\s+(?:with\s+)?(?:gem|folfir|folfox|abraxane|capecitabine|chemo)",
        source,
    )
    chemotherapy = re.search(
        r"folfirinox|mfolfirinox|folfox|folfiri|gemcitabine|\bgem\b|gemzar|"
        r"abraxane|nab-paclitaxel|capecitabine|xeloda|5-fu|nal-iri|chemotherapy",
        current_meds,
        re.IGNORECASE,
    )
    if held and chemotherapy and not active_now:
        return "", "systemic anticancer therapy explicitly held/paused"
    return current_meds, None


_CURRENT_MED_PATTERNS = (
    ("FOLFIRINOX", re.compile(r"\bm?folfirinox\b", re.IGNORECASE)),
    ("FOLFOX", re.compile(r"\bm?folfox\b", re.IGNORECASE)),
    ("FOLFIRI", re.compile(r"\bm?folfiri\b", re.IGNORECASE)),
    ("gemcitabine", re.compile(r"\b(?:gemcitabine|gemzar|gem)\b", re.IGNORECASE)),
    ("abraxane", re.compile(r"\b(?:abraxane|nab[\s-]?paclitaxel)\b", re.IGNORECASE)),
    ("capecitabine", re.compile(r"\b(?:capecitabine|xeloda|cape)\b", re.IGNORECASE)),
    ("irinotecan", re.compile(r"\birinotecan\b", re.IGNORECASE)),
    ("oxaliplatin", re.compile(r"\boxaliplatin\b", re.IGNORECASE)),
    ("5-FU", re.compile(r"\b(?:5[\s-]?fu|fluorouracil)\b", re.IGNORECASE)),
    ("leucovorin", re.compile(r"\bleucovorin\b", re.IGNORECASE)),
)

_CHEMOTHERAPY_MED_IDS = {
    "FOLFIRINOX", "FOLFOX", "FOLFIRI", "gemcitabine", "abraxane",
    "capecitabine", "irinotecan", "oxaliplatin", "5-FU", "leucovorin",
}


def _current_med_hits(text):
    """Return ``(start, end, canonical)`` medication hits in textual order."""
    text = str(text or "")
    hits = []
    for canonical, pattern in _CURRENT_MED_PATTERNS:
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), canonical))
    hits.sort(key=lambda item: item[0])
    return hits


def _current_med_ids(text):
    """Return supported canonical regimen/drug names in textual order."""
    return list(dict.fromkeys(canonical for _, _, canonical in _current_med_hits(text)))


def _current_med_items(current_meds):
    """Normalize known regimens while conservatively retaining unknown extracted items."""
    items = []
    for raw in re.split(r"[,;\n]", str(current_meds or "")):
        raw = raw.strip()
        if not raw:
            continue
        known = _current_med_ids(raw)
        if known:
            items.extend((name, name, True) for name in known)
        else:
            items.append((raw.lower(), raw, False))
    deduped = []
    seen = set()
    for identity, label, known in items:
        key = identity.lower()
        if key not in seen:
            seen.add(key)
            deduped.append((identity, label, known))
    return deduped


def resolve_current_anticancer_meds(
    current_meds,
    note_text,
    assessment_and_plan,
    recent_changes="",
):
    """Conservatively reconcile the final active anticancer regimen.

    This helper is intentionally narrower than a medication extractor.  It can recover a
    small set of standard regimens from strong current-administration language, remove a
    clearly superseded/held/completed/planned regimen, and otherwise preserves unknown
    extracted drug names.  A one-dose or one-cycle delay does not end an active regimen.
    """
    note_text = str(note_text or "")
    assessment_and_plan = str(assessment_and_plan or "")
    recent_changes = str(recent_changes or "")
    current_meds = str(current_meds or "").strip()
    reasons = []

    items = _current_med_items(current_meds)
    unknown_items = [(identity, label) for identity, label, known in items if not known]

    sources = (
        ("ap", assessment_and_plan),
        ("changes", recent_changes),
        ("note", note_text),
    )
    active_ids = []
    tentative_ids = []
    planned_ids = set()
    past_ids = set()
    stopped_ids = set()

    literature_re = re.compile(
        r"\b(?:patients? (?:receiving|treated)|study|trial|phase [i1-3]+|"
        r"response rate|published|et al)\b",
        re.IGNORECASE,
    )
    plan_re = re.compile(
        r"\b(?:recommend(?:ed)?|consider(?:ed|ing)?|option(?:s)?|candidate|could|may|"
        r"plan(?:ned)? to|will start|would start|if (?:she|he|the patient)|if .* opts?)\b",
        re.IGNORECASE,
    )
    past_re = re.compile(
        r"\b(?:previously|prior|histor(?:y|ical)|completed|finished|s/p|status post|"
        r"progress(?:ed|ion) on|after completing|had received)\b",
        re.IGNORECASE,
    )
    stopped_re = re.compile(
        r"\b(?:stopped|discontinued|d/c(?:'d|ed)?|no longer|hold|held|holding|permanently omitted|"
        r"not taking|omitted(?:\s+\w+){0,3}\s+since|switched off)\b",
        re.IGNORECASE,
    )
    strong_active_re = re.compile(
        r"\b(?:currently|now|still)\s+(?:on|receiving|treated with)|"
        r"\b(?:continue(?:s|d|ing)?|remain(?:s|ed)? on|receiving|being given|"
        r"on treatment with|on therapy with)\b|"
        r"\b(?:cycle\s*#?\s*\d+|c\d+\s*d\d+|presents? for\s+c\d+)|"
        r"\b(?:received|administered|was given|began|initiated|started)\b[^.;\n]{0,80}"
        r"\b(?:today|this visit)\b|"
        r"\b(?:today|this visit)\b[^.;\n]{0,50}\b(?:received|administered|was given)\b",
        re.IGNORECASE,
    )
    started_re = re.compile(r"\b(?:started|began|initiated)\b", re.IGNORECASE)
    schedule_switch_re = re.compile(
        r"\bswitch(?:ed)?\b[^.;\n]{0,100}\b(?:alternate|every other|weekly|biweekly)\b",
        re.IGNORECASE,
    )
    note_current_anchor_re = re.compile(
        r"\b(?:current(?:ly)?|today|this visit|presents? for|continues?|continuing|"
        r"most recently|now (?:on|receiving|treated|at|had)|remains? on)\b",
        re.IGNORECASE,
    )

    # Sentence-level classification prevents an active keyword elsewhere in a long note from
    # converting historical or trial regimens into current medication.
    for source_name, source_text in sources:
        clauses = re.split(
            r"[;\n]+|(?<=[.!?])\s+|\s{2,}(?=[#-])",
            str(source_text or ""),
        )
        for sentence in (clause.strip() for clause in clauses if clause.strip()):
            ids = _current_med_ids(sentence)
            if not ids or literature_re.search(sentence):
                continue
            is_planned = bool(plan_re.search(sentence))
            is_past = bool(past_re.search(sentence))
            is_stopped = bool(stopped_re.search(sentence))
            is_active = bool(strong_active_re.search(sentence))
            if source_name == "note" and is_active and not note_current_anchor_re.search(sentence):
                # A bare C1D1/C2D1 in a longitudinal treatment timeline is historical, not proof
                # that the regimen is active at this visit.
                is_active = False
            if is_planned and not re.search(r"\bwill continue\b", sentence, re.IGNORECASE):
                for match in plan_re.finditer(sentence):
                    planned_ids.update(
                        _current_med_ids(sentence[max(0, match.start() - 45):match.end() + 80])
                    )
            if is_past:
                past_ids.update(ids)
            if is_stopped:
                for match in stopped_re.finditer(sentence):
                    nearby = _current_med_hits(
                        sentence[max(0, match.start() - 35):match.end() + 45]
                    )
                    if nearby:
                        window_start = max(0, match.start() - 35)
                        local_start = match.start() - window_start
                        local_end = match.end() - window_start
                        nearest = min(
                            nearby,
                            key=lambda hit: min(
                                abs(hit[0] - local_end), abs(local_start - hit[1])
                            ),
                        )
                        stopped_ids.add(nearest[2])
            if is_active and not is_planned and not is_stopped and not is_past:
                active_ids.extend(ids)
            elif (started_re.search(sentence) or schedule_switch_re.search(sentence)) \
                    and not is_planned and not is_stopped and not is_past:
                tentative_ids.extend(ids)

    active_ids = list(dict.fromkeys(active_ids))
    tentative_ids = list(dict.fromkeys(tentative_ids))

    combined_current = f"{assessment_and_plan}\n{recent_changes}"
    all_source = f"{combined_current}\n{note_text}"
    generic_continue = re.search(
        r"\b(?:will\s+)?continue(?:\s+on|\s+with)?\s+(?:the\s+)?(?:current\s+)?treatment\b|"
        r"\bpresents?\s+for\s+c\d+(?:d\d+)?\b",
        combined_current,
        re.IGNORECASE,
    )
    note_generic_continue = re.search(
        r"\btolerat(?:e|es|ed|ing)\s+(?:the\s+)?current\s+chemo(?:therapy)?\b|"
        r"\bnow\s+(?:has|had)\s+\d+\s+(?:full\s+)?cycles?\b|"
        r"\bmost recently (?:received|completed)\b[^.;\n]{0,40}\bcycle\b",
        note_text,
        re.IGNORECASE,
    )
    generic_continue = generic_continue or note_generic_continue
    if generic_continue and not active_ids:
        viable = [
            name for name in tentative_ids
            if name not in stopped_ids and name not in past_ids
        ]
        # Bind an unnamed "continue current treatment" only to one unambiguous regimen family.
        # Gemcitabine + Abraxane and Gemcitabine + capecitabine count as one standard doublet.
        viable_set = set(viable)
        known_doublet = (
            viable_set in ({"gemcitabine", "abraxane"}, {"gemcitabine", "capecitabine"})
        )
        if len(viable_set) == 1 or known_doublet:
            active_ids.extend(viable)
            reasons.append("bound unnamed continuation to the only viable recent regimen")

    # In a current A/P, "s/p N cycles of X" immediately followed by "presents for C(N+1)"
    # or "continue current treatment" describes an ongoing course.  The same phrase in the
    # full-note longitudinal timeline is intentionally not used.
    ap_sentences = _clinical_sentences(assessment_and_plan)
    for index, sentence in enumerate(ap_sentences):
        if not re.search(r"\b(?:presents? for\s+c\d+|continue(?:s|d|ing)?\s+(?:the\s+)?current treatment)\b", sentence, re.I):
            continue
        for prior in ap_sentences[max(0, index - 1):index + 1]:
            if re.search(r"\bs/p\s+\d+\s+cycles?\b", prior, re.I) \
                    and not re.search(r"\b(?:completed|finished|holiday|break)\b", prior, re.I):
                active_ids.extend(_current_med_ids(prior))
    active_ids = list(dict.fromkeys(active_ids))

    # A named current regimen followed by "only" or "going forward" supersedes older regimens.
    exclusive_ids = []
    for canonical, pattern in _CURRENT_MED_PATTERNS:
        current_only = re.search(
            r"\b(?:continue(?:s|d|ing)?|currently\s+(?:on|receiving)|remain(?:s|ed)? on)\b"
            r"[^.;\n]{0,35}" + pattern.pattern + r"[^.;\n]{0,15}\b(?:only|going forward)\b",
            combined_current,
            re.IGNORECASE,
        )
        switched_to = re.search(
            r"\bswitch(?:ed|ing)?\s+to\b[^.;\n]{0,25}" + pattern.pattern,
            combined_current,
            re.IGNORECASE,
        )
        if current_only or switched_to:
            exclusive_ids.append(canonical)
    exclusive_ids = list(dict.fromkeys(exclusive_ids))
    if exclusive_ids:
        active_ids = list(dict.fromkeys(exclusive_ids + active_ids))
        reasons.append("current exclusive regimen overrides historical regimen")

    # Irinotecan omission is the clinically decisive distinction between historical
    # FOLFIRINOX and current FOLFOX in a common toxicity-driven switch.
    irinotecan_omitted = re.search(
        r"\b(?:omit(?:ted|ting)?|stop(?:ped|ping)?|discontinu(?:ed|ing))\b"
        r"[^.;\n]{0,35}\birinotecan\b|"
        r"\birinotecan\b[^.;\n]{0,35}\b(?:omit(?:ted)?|stop(?:ped)?|discontinu(?:ed)?)\b",
        combined_current,
        re.IGNORECASE,
    )
    if irinotecan_omitted:
        stopped_ids.add("irinotecan")
        if "FOLFOX" in active_ids or "FOLFOX" in exclusive_ids:
            stopped_ids.add("FOLFIRINOX")

    single_delay = re.search(
        r"\b(?:today'?s?|this)\s+(?:infusion|dose|cycle)\b[^.;\n]{0,55}"
        r"\b(?:postpon(?:e|ed)|delay(?:ed)?|cancel(?:led|ed)?|skip(?:ped)?|hold|held)\b|"
        r"\b(?:postpon(?:e|ed)|delay(?:ed)?|cancel(?:led|ed)?|skip(?:ped)?)\b"
        r"[^.;\n]{0,55}\b(?:today'?s?|this)\s+(?:infusion|dose|cycle)\b|"
        r"\b(?:day\s*\d+|c\d+(?:d\d+)?|cycle\s*#?\s*\d+)\b[^.;\n]{0,55}"
        r"\b(?:postpon(?:e|ed)|delay(?:ed)?|cancel(?:led|ed)?|skip(?:ped)?)\b",
        combined_current,
        re.IGNORECASE,
    )
    actual_today = re.search(
        r"\b(?:received|administered|was given|proceed(?:ed)? with)\b[^.;\n]{0,80}"
        r"\b(?:today|this visit)\b|"
        r"\b(?:today|this visit)\b[^.;\n]{0,50}\b(?:received|administered|was given)\b",
        combined_current,
        re.IGNORECASE,
    )
    global_hold = re.search(
        r"\b(?:hold|held|holding|pause|paused)\b[^.;\n]{0,35}"
        r"\b(?:all\s+)?(?:chemo(?:therapy)?|systemic therapy|treatment|regimen)\b|"
        r"\b(?:chemo(?:therapy)?|systemic therapy|treatment|regimen)\b[^.;\n]{0,35}"
        r"\b(?:on hold|held|paused)\b",
        combined_current,
        re.IGNORECASE,
    )
    completed_break = False
    for sentence in _clinical_sentences(all_source):
        if plan_re.search(sentence) and not re.search(r"\bcurrently|\bnow\b", sentence, re.I):
            continue
        completed_course = re.search(
            r"\b(?:completed|finished)\b[^.;\n]{0,55}\b(?:cycles?|chemo(?:therapy)?|treatment)\b",
            sentence,
            re.IGNORECASE,
        )
        inactive_state = re.search(
            r"\b(?:now|currently)\b[^.;\n]{0,30}\b(?:on|under)\b[^.;\n]{0,15}"
            r"\b(?:chemo(?:therapy)?|treatment)?\s*(?:break|holiday|surveillance|observation)\b|"
            r"\b(?:chemo(?:therapy)?|treatment)\s+(?:break|holiday)\b|"
            r"\bno\s+(?:current|active)\s+(?:or\s+future\s+)?chemo(?:therapy)?\b",
            sentence,
            re.IGNORECASE,
        )
        if completed_course and re.search(r"\b(?:break|holiday|surveillance|observation)\b", sentence, re.I):
            completed_break = True
            break
        if inactive_state:
            completed_break = True
            break
    never_started = re.search(
        r"\b(?:no treatment has (?:yet )?started|has not started (?:any )?(?:cancer )?treatment|"
        r"not yet (?:on|started) (?:any )?(?:cancer )?treatment|treatment[- ]naive)\b",
        combined_current,
        re.IGNORECASE,
    )

    kept_known = []
    existing_ids = [identity for identity, _, known in items if known]
    existing_id_set = set(existing_ids)
    for identity in existing_ids:
        if identity in stopped_ids and identity not in active_ids:
            reasons.append(f"removed stopped/superseded {identity}")
            continue
        if exclusive_ids and identity not in exclusive_ids:
            reasons.append(f"removed non-current regimen {identity}")
            continue
        if never_started and identity not in active_ids:
            reasons.append(f"removed planned-only {identity}")
            continue
        if identity in planned_ids and identity not in active_ids:
            reasons.append(f"removed planned-only {identity}")
            continue
        if completed_break and identity in _CHEMOTHERAPY_MED_IDS and identity not in active_ids:
            reasons.append(f"removed completed/on-break {identity}")
            continue
        if global_hold and not single_delay and not actual_today \
                and identity in _CHEMOTHERAPY_MED_IDS:
            reasons.append(f"removed held {identity}")
            continue
        kept_known.append(identity)

    # When the prior output was empty or incomplete, add only source-grounded active regimens.
    if not (global_hold and not single_delay and not actual_today) and not completed_break and not never_started:
        for identity in active_ids:
            if identity in stopped_ids and identity not in active_ids:
                continue
            if exclusive_ids and identity not in exclusive_ids:
                continue
            if existing_id_set and identity not in existing_id_set and not exclusive_ids:
                # With a non-empty extracted regimen, do not append a different regimen merely
                # because a long note contains another active-looking historical passage.  The
                # sole exception is completing a source-confirmed standard doublet.
                pair_completion = any(
                    identity in pair and bool(existing_id_set.intersection(pair))
                    for pair in (
                        {"gemcitabine", "abraxane"},
                        {"gemcitabine", "capecitabine"},
                    )
                )
                if not pair_completion:
                    continue
            if identity not in kept_known:
                kept_known.append(identity)
                reasons.append(f"added active {identity}")

    # Planned-only known drugs can be removed, but unfamiliar extracted names are preserved unless
    # the source gives a global no-treatment/hold state.  This is the main anti-overreach guard.
    kept_unknown = [label for _, label in unknown_items]
    if never_started:
        kept_unknown = []
        if unknown_items:
            reasons.append("cleared unrecognized medication because treatment has not started")
    elif global_hold and not single_delay and not actual_today:
        # "chemotherapy held" does not justify deleting a concurrent endocrine/targeted drug whose
        # class is unknown to this deliberately small resolver.
        pass

    if set(kept_known) == existing_id_set \
            and kept_unknown == [label for _, label in unknown_items]:
        # Preserve the model's harmless casing, ordering, and regimen wording when the semantic
        # medication set did not change (e.g. "Gemcitabine + Abraxane" or "nal-IRI").
        resolved = current_meds
    else:
        resolved = ", ".join(kept_known + kept_unknown)
    return resolved, tuple(dict.fromkeys(reasons))


def sanitize_response_assessment(
    response,
    note_text,
    assessment_and_plan,
    current_meds="",
    recent_changes="",
    findings="",
):
    """Apply only high-confidence, source-grounded response-assessment corrections.

    Returns ``(cleaned_response, reasons)``.  The helper deliberately does not attempt
    general regimen/date reasoning; ambiguous cross-regimen attribution remains a prompt task.
    """
    original = str(response or "").strip()
    cleaned = original
    reasons = []
    current_response_source = "\n".join(
        str(value or "") for value in (assessment_and_plan, findings)
    )
    source = f"{current_response_source}\n{note_text or ''}"

    # The current A/P is the highest-priority response statement.  Do not let an old
    # suspicious lesion elsewhere in the note turn an explicit stable/good-control
    # assessment into progression (notably in surveillance notes).
    if _has_affirmative_progression(cleaned):
        stable_sentence = _first_supported_sentence(
            (assessment_and_plan,),
            lambda sentence: bool(
                re.search(
                    r"\b(?:continued\s+good\s+disease\s+control|good\s+disease\s+control|"
                    r"stable\s+disease|favorable\s+treatment\s+response|"
                    r"radiographic\s+evidence\s+of\s+response|responding\s+to\b|"
                    r"no\s+evidence\s+of\s+(?:recurrence|metastatic\s+disease))\b",
                    sentence,
                    re.IGNORECASE,
                )
                and not re.search(
                    r"\b(?:progress(?:ed|ion|ing)?|worsen\w*|declin\w*|new\s+metasta\w*)\b",
                    sentence,
                    re.IGNORECASE,
                )
            ),
        )
        current_ap_has_progression = _has_affirmative_progression(assessment_and_plan)
        if stable_sentence and not current_ap_has_progression:
            cleaned = stable_sentence
            reasons.append("current A/P stable-control statement overrides unsupported progression")

    # A treatment-era, explicitly favorable current assessment is more relevant than
    # a pre-treatment progression sentence copied from the longitudinal history/header.
    if _has_affirmative_progression(cleaned):
        early_response_sentence = _first_supported_sentence(
            (assessment_and_plan,),
            lambda sentence: bool(re.search(
                r"\b(?:pain|mass|symptoms?)\b[^.;]{0,80}\bimprov\w*\b"
                r"[^.;]{0,100}\b(?:early\s+)?treatment\s+response\b|"
                r"\bhopeful\s+for\s+(?:an?\s+)?(?:early\s+)?treatment\s+response\b",
                sentence,
                re.IGNORECASE,
            )),
        )
        if early_response_sentence and str(current_meds or "").strip():
            cleaned = early_response_sentence
            reasons.append("current treatment-era improvement overrides pre-treatment progression")

    claims_current_treatment = bool(
        re.search(
            r"\b(?:currently\s+)?(?:on|receiving|undergoing)\s+"
            r"(?:current\s+)?(?:treatment|therapy|chemotherapy)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    if claims_current_treatment and not str(current_meds or "").strip():
        state_source = f"{recent_changes or ''}\n{assessment_and_plan or ''}"
        planned = re.search(
            r"\b(?:will|plan(?:ned)? to|recommend(?:ed)?|consider|may|can|prescri\w*|order\w*)\b"
            r"[^.;\n]{0,35}\b(?:start|begin|initiat)|"
            r"\b(?:start|begin|initiat)\b[^.;\n]{0,50}\b(?:next|after|once|when)\b",
            state_source,
            re.IGNORECASE,
        )
        held_or_completed = re.search(
            r"\b(?:hold|held|holding|pause|paused|completed|finished)\b[^.;\n]{0,60}"
            r"(?:chemo|treatment|therapy|cycle)|"
            r"\b(?:chemo(?:therapy)?|systemic therapy|treatment)\s+"
            r"(?:is\s+)?(?:currently\s+)?(?:on\s+)?(?:hold|paused)|"
            r"\b(?:chemo(?:therapy)?|treatment)\s+(?:break|holiday)\b",
            state_source,
            re.IGNORECASE,
        )
        if planned and not held_or_completed:
            cleaned = "Not yet on treatment — no response to assess."
            reasons.append("empty final meds plus planned-only treatment")
        elif held_or_completed:
            cleaned = _direct_tumor_status_sentence(
                assessment_and_plan, findings, note_text
            ) or "Not mentioned in note."
            reasons.append("empty final meds plus held/completed treatment")

    was_untreated = bool(UNTREATED_RESPONSE_RE.search(cleaned))
    recurrence_sentence = ""
    if was_untreated:
        recurrence_sentence = _positive_recurrence_or_growth_sentence(
            assessment_and_plan, findings, note_text
        )
        if recurrence_sentence:
            cleaned = recurrence_sentence
            reasons.append("affirmative recurrence/growth replaces untreated response")

    if was_untreated and _same_day_treatment_started(
        assessment_and_plan, recent_changes, current_meds
    ):
        start_statement = "Anticancer treatment started today; too early to assess its response."
        cleaned = f"{cleaned} {start_statement}" if recurrence_sentence else start_statement
        reasons.append("same-day treatment start")

    if re.search(r"\bpartial response\b", cleaned, re.IGNORECASE) and not re.search(
        r"\bpartial (?:radiographic |pathologic )?response\b",
        current_response_source,
        re.IGNORECASE,
    ):
        decrease_sentence = _first_supported_sentence(
            (assessment_and_plan, findings, note_text),
            lambda sentence: bool(
                re.search(r"\b(?:slight|minimal|mild)(?:ly)?\b", sentence, re.IGNORECASE)
                and re.search(r"\b(?:decreas\w*|reduc\w*|smaller|shrink\w*)\b", sentence, re.IGNORECASE)
                and TUMOR_RESPONSE_ANCHOR_RE.search(sentence)
            ),
        )
        if decrease_sentence:
            if _has_affirmative_progression(cleaned):
                degree = re.search(
                    r"\b(slight|minimal|mild)(?:ly)?\s+(decreas\w*|reduc\w*|shrink\w*)",
                    decrease_sentence,
                    re.IGNORECASE,
                )
                replacement = (
                    f"{degree.group(1).lower()} decrease" if degree else "documented decrease"
                )
                cleaned = re.sub(
                    r"\b(?:a\s+)?partial response\b",
                    replacement,
                    cleaned,
                    flags=re.IGNORECASE,
                )
            else:
                cleaned = decrease_sentence
            reasons.append("unsupported formal partial-response label removed")

    disappearance = _explicit_treatment_linked_mass_disappearance(
        assessment_and_plan, findings, note_text
    )
    if disappearance and "no longer" not in cleaned.lower():
        if _has_affirmative_progression(cleaned):
            cleaned = f"{cleaned.rstrip()} {disappearance}"
            reasons.append("progression and treatment-linked mass disappearance both preserved")
        else:
            cleaned = disappearance
            reasons.append("explicit treatment-linked mass disappearance preserved")

    sentences = _clinical_sentences(cleaned)
    if sentences:
        surgery_context = bool(
            re.search(
                r"\b(?:status post|s/p|underwent|prior)\b[^.;\n]{0,45}"
                r"(?:resection|surgery|whipple|pancreatectomy|mastectomy|lumpectomy)|"
                r"post[\s-]*(?:operative|surgical)|postsurgical changes",
                source,
                re.IGNORECASE,
            )
        )
        kept = []
        dropped = False
        for sentence in sentences:
            lower = sentence.lower()
            pure_non_response = bool(
                NON_RESPONSE_FINDING_RE.search(sentence)
                and not TUMOR_RESPONSE_ANCHOR_RE.search(sentence)
            )
            pure_postop_vascular = bool(
                surgery_context
                and re.search(
                    r"portal vein|superior mesenteric vein|vascular narrowing|vascular occlusion",
                    lower,
                )
                and not TUMOR_RESPONSE_ANCHOR_RE.search(sentence)
            )
            if pure_non_response or pure_postop_vascular:
                dropped = True
                continue
            kept.append(sentence)
        if dropped:
            if kept:
                cleaned = " ".join(kept)
            else:
                cleaned = _direct_tumor_status_sentence(
                    assessment_and_plan, findings, note_text
                ) or "Not assessed at this visit."
            reasons.append("pure toxicity/postoperative/biliary finding removed")

    cleaned = cleaned.strip()
    return cleaned, tuple(reasons)


def recover_completed_genetic_results(value, note_text):
    """Add only unmistakably completed MMR/MSI results omitted by generation."""
    cleaned, reasons = sanitize_genetic_testing_results(value)
    note = str(note_text or "")
    recovered = []

    mmr_intact = bool(re.search(
        r"\bMMR\s+proteins?\s+(?:all\s+)?intact\b(?:\s+by\s+IHC)?",
        note,
        re.IGNORECASE,
    ))
    if not mmr_intact:
        mmr_intact = all(re.search(
            rf"\b{gene}\s+expression\s*:\s*Present\b", note, re.IGNORECASE
        ) for gene in ("MLH1", "PMS2", "MSH2", "MSH6"))
    if mmr_intact and not re.search(
        r"\b(?:MMR|mismatch\s+repair|pMMR|MLH1|PMS2|MSH2|MSH6)\b",
        cleaned,
        re.IGNORECASE,
    ):
        recovered.append("MMR proteins intact by IHC (pMMR)")

    if not recovered:
        return cleaned, reasons
    existing = "" if cleaned == GENETIC_RESULTS_FALLBACK else cleaned.rstrip(". ")
    merged = "; ".join(part for part in (existing, *recovered) if part).rstrip(". ") + "."
    return merged, tuple(dict.fromkeys((*reasons, "recovered completed MMR result from source")))


def sanitize_breast_recurrence_receptors(value, assessment_and_plan, note_text=""):
    """Prevent unsupported HER2 claims and cross-timepoint receptor borrowing."""
    original = str(value or "").strip()
    ap = str(assessment_and_plan or "")
    source = f"{ap}\n{note_text or ''}"
    reasons = []

    her2_supported = bool(re.search(
        r"\bHER\s*-?\s*2\b|\bERBB2\b|\btriple[\s-]*negative\b|\bTNBC\b|"
        r"\bFISH\b|\b(?:trastuzumab|pertuzumab|herceptin|perjeta|kadcyla|enhertu)\b|"
        r"\b(?:ER|PR|receptor)\b[^.;\n]{0,30}\*{3,}\s*-",
        source,
        re.IGNORECASE,
    ))
    if original and not her2_supported:
        stripped = re.sub(
            r"(?i)(?:\s*[/,]\s*)?HER\s*-?\s*2\s*(?:status\s*)?"
            r"(?:[:=]?\s*)?(?:positive|negative|pos\b|neg\b|[+-])",
            "",
            original,
        )
        stripped = re.sub(r"/{2,}", "/", stripped)
        stripped = re.sub(r"\s{2,}", " ", stripped).strip(" /,;")
        if stripped != original:
            original = stripped
            reasons.append("removed HER2 claim absent from source")

    if not original or not re.search(
        r"\b(?:local(?:ly)?\s+)?recurr\w*\b[^.;]{0,120}\b(?:strongly\s+)?"
        r"hormone[\s-]*receptor\s+positive\b|"
        r"\b(?:strongly\s+)?hormone[\s-]*receptor\s+positive\b[^.;]{0,120}\brecurr\w*\b",
        ap,
        re.IGNORECASE,
    ):
        return original, tuple(reasons)

    # If the current recurrence sentence itself gives PR or HER2, retain the generated profile.
    current_sentences = [
        sentence for sentence in _clinical_sentences(ap)
        if re.search(r"\brecurr\w*\b", sentence, re.IGNORECASE)
    ]
    current_context = " ".join(current_sentences)
    if re.search(
        r"\b(?:PR|progesterone\s+receptor|HER\s*-?\s*2)\b\s*"
        r"(?:[:=]?\s*(?:positive|negative|pos\b|neg\b|[+-]|\d{1,3}\s*%))",
        current_context,
        re.IGNORECASE,
    ):
        return original, tuple(reasons)

    clauses = [part.strip() for part in original.split(";")]
    changed = False
    for index, clause in enumerate(clauses):
        if re.search(r"\b(?:current|recurrent|recurrence)\b", clause, re.IGNORECASE) and re.search(
            r"\b(?:ER|PR|HER\s*-?\s*2)\b", clause, re.IGNORECASE
        ):
            label = "current recurrent disease"
            match = re.search(r"\(([^)]*\b(?:current|recurrent|recurrence)\b[^)]*)\)", clause, re.IGNORECASE)
            if match:
                label = match.group(1)
            clauses[index] = f"HR+ (PR/HER2 not specified; {label})"
            changed = True
    if not changed:
        return original, tuple(reasons)
    reasons.append("removed unsupported recurrent-lesion PR/HER2 borrowed from history")
    return "; ".join(clauses), tuple(reasons)


GENETIC_RESULTS_FALLBACK = "No genetic testing results in note."

_GENETIC_RESULT_ANCHOR_RE = re.compile(
    r"\b(?:mammaprint|oncotype(?:\s+dx)?|recurrence\s+score|"
    r"foundation(?:one)?|strata|ucsf\s*500|tempus|guardant|invitae|ambry|"
    r"germ\s*line|germline|somatic|molecular\s+profil\w*|gene(?:tic)?\s+panel|"
    r"sequenc\w*|ngs|ctdna|liquid\s+biopsy|"
    r"mmr|mismatch\s+repair|msi|mss|microsatellite|tmb|tumou?r\s+mutational|"
    r"hrd|homologous\s+recombination|pd[\s-]?l1|cps|"
    r"mutation|mutated|variant|vus|pathogenic|carrier|"
    r"brca\s*[12]?|atm|palb2|chek2|mlh1|msh2|msh6|pms2|epcam|lynch|"
    r"kras|k-ras|tp53|p53|pik3ca|braf|ntrk|esr1|egfr|alk|ros1|"
    r"cdkn2a|cdkn2b|smad4|spink1|rb1|fan[ca-z0-9]+|nf2|axin1|ctc1|"
    r"ercc4|mc1r|recql4|apc|mtor|nkx2-1|pdgfrb|pik3c2g|tnfaip3|cbl|"
    r"erbb2\s+(?:amplification|amplified|mutation|variant)|"
    r"ca\s*19-?9\s+non-?secretor|non-?secretor)\b",
    re.IGNORECASE,
)

_GENETIC_PENDING_RE = re.compile(
    r"\b(?:pending|ordered|sent|submitted|planned|in[\s-]*process|"
    r"awaiting(?:\s+results?)?|not\s+yet\s+resulted|to\s+be\s+(?:sent|ordered|done)|"
    r"will\s+be\s+(?:sent|ordered|done))\b",
    re.IGNORECASE,
)

_GENETIC_COMPLETED_RE = re.compile(
    r"\b(?:result(?:ed|s)?|show(?:ed|s)|found|identified|detected|harbou?rs?|"
    r"positive|negative|pathogenic|benign|likely\s+pathogenic|variant|vus|mutation|"
    r"carrier|amplified|amplification|intact|deficient|loss\s+of|stable|unstable|"
    r"high[\s-]*risk|low[\s-]*risk|score\s*[:=]?\s*\d|cps\s*[:=]?\s*\d|"
    r"tmb\s*[:=]?\s*\d|\d+\s*muts?/?mb|undetermined|no\s+actionable)\b",
    re.IGNORECASE,
)

_RELATIVE_RE = re.compile(
    r"\b(?:mother|father|brother|sister|daughter|son|aunt|uncle|cousin|"
    r"grandmother|grandfather|relative|family\s+member)\b",
    re.IGNORECASE,
)

_PATIENT_SELF_RE = re.compile(
    r"\b(?:patient(?!['’]s)|she|he|her\s+(?:germline|tumou?r|testing)|"
    r"his\s+(?:germline|tumou?r|testing))\b",
    re.IGNORECASE,
)

_ROUTINE_BREAST_RECEPTOR_RE = re.compile(
    r"\b(?:er|estrogen\s+receptors?|pr|progesterone\s+receptors?|"
    r"her\s*-?\s*2|ki\s*-?\s*67)\b",
    re.IGNORECASE,
)

_SURGICAL_PATHOLOGY_RE = re.compile(
    r"\b(?:surgical\s+pathology|outside\s+path|pathology|did\s+not\s+repeat\s+markers|"
    r"no\s+grade|invasive\s+(?:ductal|lobular|mammary)|"
    r"ductal\s+carcinoma|lobular\s+carcinoma|dcis|lcis|lymphovascular|lvi|"
    r"margin(?:s)?|sentinel|micrometasta\w*|extranodal|extracapsular|"
    r"lymph\s+nodes?|\d+\s*/\s*\d+\s+(?:nodes?|ln)|tumou?r\s+size|"
    r"grade\s*[1-3]|necrosis|p[ty]?t\d|p[ny]?n\d)\b",
    re.IGNORECASE,
)


def _split_genetic_result_clauses(value):
    """Split only on strong independent-clause boundaries, not commas or decimals."""
    text = str(value or "").strip()
    text = re.sub(r"\s*[•▪]\s*", "\n", text)
    clauses = re.split(
        r"\s*(?:;|\n+)\s*|(?<=[.!?])\s+(?=(?:[A-Z+\[]|\d{1,2}/\d{1,2}/\d{2,4}))",
        text,
    )
    return [clause.strip() for clause in clauses if clause.strip()]


def sanitize_genetic_testing_results(value):
    """Remove only high-confidence non-results from ``genetic_testing_results``.

    The helper deliberately does not search the source note or add missing results.  It removes
    independently delimited family-member results, unfinished tests, routine breast receptor
    pathology, and pure surgical pathology while preserving completed molecular assays.  In
    particular, MMR/PD-L1 IHC, molecular ERBB2 amplification, and CA 19-9 non-secretor status are
    valid results and must survive.

    Returns ``(cleaned_value, reasons)``.
    """
    original = str(value or "").strip()
    normalized_empty = original.lower().rstrip(". ")
    if normalized_empty in (
        "", "none", "n/a", "na", "not available", "not mentioned",
        "no genetic testing results in note",
    ):
        if original == GENETIC_RESULTS_FALLBACK:
            return original, ()
        return GENETIC_RESULTS_FALLBACK, ("normalized empty fallback",)

    kept = []
    reasons = []
    changed = False
    for clause in _split_genetic_result_clauses(original):
        bare = clause.strip().rstrip(".; ")
        if not bare:
            continue
        has_genetic_anchor = bool(_GENETIC_RESULT_ANCHOR_RE.search(bare))

        # A family member's result is not the patient's result.  If the same clause explicitly
        # contains a patient result too, preserve it rather than risk deleting valid information;
        # the extraction prompt must separate such mixed prose into independent clauses.
        if _RELATIVE_RE.search(bare) and has_genetic_anchor \
                and not _PATIENT_SELF_RE.search(bare):
            reasons.append("removed family-member result")
            changed = True
            continue

        # Sent/ordered/pending assays are plans, not completed results.  A clause that also contains
        # an explicit completed-result marker is retained conservatively.
        if _GENETIC_PENDING_RE.search(bare) and not _GENETIC_COMPLETED_RE.search(bare):
            reasons.append("removed pending or ordered test")
            changed = True
            continue

        valid_ihc_context = bool(re.search(
            r"\b(?:mmr|mismatch\s+repair|mlh1|msh2|msh6|pms2|pd[\s-]?l1|cps)\b",
            bare,
            re.IGNORECASE,
        ))
        valid_erbb2_context = bool(re.search(
            r"\b(?:foundation(?:one)?|strata|ucsf\s*500|tempus|guardant|ngs|"
            r"sequenc\w*|molecular\s+profil\w*)\b[^.;]*\berbb2\b|"
            r"\berbb2\b[^.;]*\b(?:amplification|amplified|mutation|variant)\b",
            bare,
            re.IGNORECASE,
        ))
        routine_receptor = bool(_ROUTINE_BREAST_RECEPTOR_RE.search(bare))
        if routine_receptor and not valid_ihc_context and not valid_erbb2_context:
            reasons.append("removed routine ER/PR/HER2/Ki-67 pathology")
            changed = True
            continue

        if _SURGICAL_PATHOLOGY_RE.search(bare) and not has_genetic_anchor:
            reasons.append("removed pure surgical pathology")
            changed = True
            continue

        # Exact duplicate clauses are harmless but common in model output; normalize them only when
        # another cleanup has already made the value change.
        if bare.lower() not in {item.lower() for item in kept}:
            kept.append(bare)
        else:
            reasons.append("removed duplicate result")
            changed = True

    if not kept:
        return GENETIC_RESULTS_FALLBACK, tuple(dict.fromkeys(reasons))
    if not changed:
        return original, ()
    cleaned = "; ".join(kept).rstrip(".; ") + "."
    return cleaned, tuple(dict.fromkeys(reasons))
