"""
SAHAYAK-TB — Post-processing validation and correction module.

Fixes three classes of errors in LLM output:
1. Tier correction: deterministic override based on factor count + treatment month
2. Risk factor deduplication: fuzzy-match and merge duplicates
3. Citation grounding: map model-generated factors to real citations from risk_factors.json

Usage:
    from inference.postprocess import correct_output
    corrected = correct_output(raw_model_output_dict)
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Load ground-truth risk factors
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_RISK_FACTORS_PATH = _DATA_DIR / "risk_factors.json"

with open(_RISK_FACTORS_PATH) as f:
    GROUND_TRUTH_FACTORS = json.load(f)

# Build keyword index for fuzzy matching model outputs to ground truth
_FACTOR_KEYWORDS = {}
for gt in GROUND_TRUTH_FACTORS:
    keywords = set()
    # Add label words
    for word in gt["label"].lower().split():
        if len(word) > 3:
            keywords.add(word)
    # Add factor id parts
    for part in gt["factor"].split("_"):
        if len(part) > 3:
            keywords.add(part)
    # Add key phrases from natural_language_variants
    for variant in gt["natural_language_variants"]:
        for word in variant.lower().split():
            if len(word) > 4:
                keywords.add(word)
    _FACTOR_KEYWORDS[gt["factor"]] = keywords


# ---------------------------------------------------------------------------
# 1. Citation grounding
# ---------------------------------------------------------------------------
def _match_factor_to_ground_truth(factor_text: str) -> dict | None:
    """Match a model-generated factor description to ground truth.
    
    Uses keyword overlap scoring to find the best match.
    Returns the ground truth entry or None if no confident match.
    """
    text_lower = factor_text.lower()
    
    # Direct keyword matching with scoring
    scores = {}
    for gt in GROUND_TRUTH_FACTORS:
        score = 0
        factor_id = gt["factor"]
        
        # Check label similarity
        label_lower = gt["label"].lower()
        if label_lower in text_lower or text_lower in label_lower:
            score += 10
        
        # Check keyword overlap
        keywords = _FACTOR_KEYWORDS[factor_id]
        for kw in keywords:
            if kw in text_lower:
                score += 1
        
        # Specific pattern matches for common model outputs
        patterns = {
            "alcohol_use": r"alcohol|sharab|drink|substance\s*use|addiction|dependence",
            "male_sex": r"\bmale\b|purush|\bman\b|\bmen\b",
            "retreatment_case": r"retreatment|category\s*ii|relapse|previously\s*treated|prior\s*default|treatment\s*failure|re-?registration",
            "migration_no_fixed_address": r"migra|no\s*(fixed|permanent)\s*address|pravasi|mobile|moves?\s*between|no\s*stable|temporary|no\s*local\s*support|relocation|moves?\s*frequently|frequent\s*relocation|shifting",
            "intensive_phase_side_effects": r"side\s*effect|nausea|vomit|joint\s*pain|rash|itching|adverse|hepato|stomach|appetite|drowsi|vision|injection\s*pain",
            "hiv_coinfection": r"\bhiv\b|\bart\b|co-?infect|antiretroviral|cd4",
            "daily_wage_poverty": r"daily\s*wage|poverty|bpl|poor|low\s*socio|wage\s*lab|income|financial|economic|afford|money|cost|expensive|funds?\s*running",
            "smoking_tobacco": r"smok|tobacco|bidi|gutka|tambaaku|cigarette",
            "drtb_long_treatment": r"dr[\-\s]*tb|mdr|drug\s*resist|extended\s*treat|longer\s*regimen|injectable|second[\-\s]line|9[\-\s]*11\s*month|bdq|lzd|bpal",
            "distance_from_dot_center": r"distance|far\s*from|travel|transport|bus\s*fare|geographic|access\s*barrier|rural|tribal|remote",
            "young_adult": r"young\s*adult|adolesc|15[\-\s]*30|college|student|youth",
            "symptom_improvement": r"symptom\s*improv|feel(s|ing)?\s*better|no\s*more\s*cough|stop\s*early|doesn.*need\s*medicine|wants?\s*to\s*stop|asks?\s*(if|about)\s*(he|she|they)?\s*can\s*stop|cessation|resolution",
            "diabetes_comorbidity": r"diabet|sugar|metformin|hba1c|blood\s*sugar|dm\b|glucose",
            "divorced_separated": r"divorc|separated|alone|no\s*family|widow|single\s*parent|isolat|no\s*support",
            "low_health_literacy": r"literacy|illiterat|doesn.*understand|confused|cannot?\s*read|picture|education|uneducated",
        }
        
        if factor_id in patterns:
            if re.search(patterns[factor_id], text_lower):
                score += 8
        
        if score > 0:
            scores[factor_id] = score
    
    if not scores:
        return None
    
    best_id = max(scores, key=scores.get)
    if scores[best_id] < 3:  # Minimum confidence threshold
        return None
    
    return next(gt for gt in GROUND_TRUTH_FACTORS if gt["factor"] == best_id)


def ground_citations(risk_factors: list[dict]) -> list[dict]:
    """Replace hallucinated citations with real ones from ground truth."""
    for rf in risk_factors:
        factor_text = rf.get("factor", "")
        gt = _match_factor_to_ground_truth(factor_text)
        if gt:
            # Replace clinical_basis with real citation
            mean_or = (gt["odds_ratio_low"] + gt["odds_ratio_high"]) / 2
            sources_str = "; ".join(gt["sources"])
            rf["clinical_basis"] = (
                f"{gt['label']} — OR {gt['odds_ratio_low']}-{gt['odds_ratio_high']} "
                f"({sources_str})"
            )
            # Store matched factor ID for deduplication
            rf["_matched_id"] = gt["factor"]
        else:
            rf["_matched_id"] = None
    return risk_factors


# ---------------------------------------------------------------------------
# 2. Risk factor deduplication
# ---------------------------------------------------------------------------
def deduplicate_factors(risk_factors: list[dict]) -> list[dict]:
    """Remove duplicate risk factors based on ground-truth matching.
    
    If two factors map to the same ground truth ID, keep the first one.
    For unmatched factors, use text similarity to detect duplicates.
    """
    seen_ids = set()
    deduplicated = []
    unmatched = []
    
    for rf in risk_factors:
        matched_id = rf.get("_matched_id")
        if matched_id:
            if matched_id not in seen_ids:
                seen_ids.add(matched_id)
                deduplicated.append(rf)
            # else: skip duplicate
        else:
            unmatched.append(rf)
    
    # For unmatched factors, check text similarity against already-added ones
    for rf in unmatched:
        is_dup = False
        for existing in deduplicated:
            similarity = SequenceMatcher(
                None,
                rf.get("factor", "").lower(),
                existing.get("factor", "").lower()
            ).ratio()
            if similarity > 0.6:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(rf)
    
    # Clean up internal field
    for rf in deduplicated:
        rf.pop("_matched_id", None)
    
    return deduplicated


# ---------------------------------------------------------------------------
# 3. Tier correction
# ---------------------------------------------------------------------------
def correct_tier(risk_factors: list[dict], treatment_month: int) -> str:
    """Deterministic tier assignment based on matched factors.
    
    Uses the same rules as the system prompt:
    - low: 0-1 factors with OR < 1.5
    - moderate: 1-2 factors with OR 1.5-2.0
    - high: 2+ factors with OR > 2.0 OR any single factor with OR > 2.5
    - critical: 3+ factors with OR > 2.0 AND treatment month > 3
    """
    if not risk_factors:
        return "low"
    
    # Map each factor back to ground truth to get OR values
    strong_factors = []  # mean OR >= 2.0
    moderate_factors = []  # mean OR 1.5-2.0
    weak_factors = []  # mean OR < 1.5
    
    # Demographic-only factors shouldn't drive tier
    demographic_ids = {"male_sex", "young_adult"}
    
    for rf in risk_factors:
        factor_text = rf.get("factor", "")
        gt = _match_factor_to_ground_truth(factor_text)
        if gt:
            if gt["factor"] in demographic_ids:
                weak_factors.append(gt)
                continue
            mean_or = (gt["odds_ratio_low"] + gt["odds_ratio_high"]) / 2
            if mean_or >= 2.0:
                strong_factors.append(gt)
            elif mean_or >= 1.5:
                moderate_factors.append(gt)
            else:
                weak_factors.append(gt)
        else:
            # Unknown factor — assume weak to avoid over-prediction
            weak_factors.append({"factor": "unknown", "odds_ratio_low": 1.2, "odds_ratio_high": 1.5})
    
    # Apply tier rules (same as compute_risk_tier in generator)
    # Critical: 3+ strong AND month > 3
    if len(strong_factors) >= 3 and treatment_month > 3:
        return "critical"
    # Critical: 2+ strong + 1+ moderate AND month > 3
    if len(strong_factors) >= 2 and len(moderate_factors) >= 1 and treatment_month > 3:
        return "critical"
    # High: 2+ strong factors
    if len(strong_factors) >= 2:
        return "high"
    # High: 1 strong + 1+ moderate
    if len(strong_factors) >= 1 and len(moderate_factors) >= 1:
        return "high"
    # High: any single factor with mean OR > 2.5
    for gt in strong_factors:
        mean_or = (gt["odds_ratio_low"] + gt["odds_ratio_high"]) / 2
        if mean_or > 2.5:
            return "high"
    # High: 1 strong + 3+ total factors
    total = len(strong_factors) + len(moderate_factors) + len(weak_factors)
    if len(strong_factors) >= 1 and total >= 3:
        return "high"
    # Moderate: 1 strong factor (OR 2.0-2.5)
    if len(strong_factors) >= 1:
        return "moderate"
    # Moderate: 2+ moderate factors
    if len(moderate_factors) >= 2:
        return "moderate"
    # Moderate: 2+ total with decent average
    all_factors_list = strong_factors + moderate_factors + weak_factors
    if len(all_factors_list) >= 2:
        avg_or = sum((f["odds_ratio_low"] + f["odds_ratio_high"]) / 2 for f in all_factors_list) / len(all_factors_list)
        if avg_or >= 1.5:
            return "moderate"
    # Low
    return "low"


# ---------------------------------------------------------------------------
# Tier boundary enforcement
# ---------------------------------------------------------------------------
def _enforce_tier_boundaries(model_tier: str, risk_factors: list[dict], treatment_month: int) -> str:
    """Enforce hard rules on tier boundaries without fully overriding the model.
    
    Only counts factors that match ground-truth evidence base.
    Unmatched factors (hallucinations like "Low weight") are ignored for tier logic.
    """
    demographic_ids = {"male_sex", "young_adult"}
    strong_count = 0
    moderate_count = 0
    matched_count = 0
    has_very_strong = False  # OR > 2.8
    
    for rf in risk_factors:
        gt = _match_factor_to_ground_truth(rf.get("factor", ""))
        if gt and gt["factor"] not in demographic_ids:
            matched_count += 1
            mean_or = (gt["odds_ratio_low"] + gt["odds_ratio_high"]) / 2
            if mean_or >= 2.0:
                strong_count += 1
                if mean_or > 2.8:
                    has_very_strong = True
            elif mean_or >= 1.5:
                moderate_count += 1
    
    # Rule 1: Critical requires month > 3 AND strong evidence
    if model_tier == "critical" and treatment_month <= 3:
        return "high"
    if model_tier == "critical" and strong_count < 2:
        return "high"
    
    # Rule 2: Upgrade high → critical when clearly warranted
    if model_tier == "high" and strong_count >= 2 and matched_count >= 3 and treatment_month > 5:
        return "critical"
    if model_tier == "high" and has_very_strong and matched_count >= 2 and treatment_month > 6:
        return "critical"
    
    # Rule 3: Upgrade moderate → high/critical
    if model_tier == "moderate" and strong_count >= 2 and treatment_month > 5:
        return "critical"
    if model_tier == "moderate" and strong_count >= 2:
        return "high"
    if model_tier == "moderate" and has_very_strong:
        return "high"
    if model_tier == "moderate" and strong_count >= 1 and matched_count >= 2:
        return "high"
    
    # Rule 4: Downgrade high → moderate if insufficient evidence
    if model_tier == "high" and strong_count == 0:
        return "moderate"
    if model_tier == "high" and strong_count == 1 and matched_count < 2:
        if not has_very_strong:
            return "moderate"
    
    # Rule 5: Downgrade moderate → low if no real factors or only weak ones
    if model_tier == "moderate" and matched_count == 0:
        return "low"
    if model_tier == "moderate" and strong_count == 0 and matched_count <= 1:
        return "low"
    
    # Rule 6: Upgrade low → moderate if strong factor present
    if model_tier == "low" and strong_count >= 1:
        return "moderate"
    
    # Otherwise trust the model
    return model_tier


# ---------------------------------------------------------------------------
# 4. Text-based factor augmentation
# ---------------------------------------------------------------------------
# High-precision patterns that can reliably extract factors from user input text.
# ONLY includes binary/unambiguous indicators that cannot appear in negative context.
# Excluded: alcohol (can be negated "nahi peeta"), poverty (can be contextually excluded),
#           distance (patient may be adherent despite distance), smoking (can be negated).
_TEXT_EXTRACTION_PATTERNS = {
    "retreatment_case": {
        "pattern": r"category\s*ii|retreatment|re-?treatment|relapse|previously\s*treated|prior\s*default|treatment\s*failure|re-?registration",
        "label": "Retreatment case (Category II)",
    },
    "intensive_phase_side_effects": {
        "pattern": r"(?<!no\s)(?<!without\s)(?<!any\s)((?:joint|body)\s*pain\s*(?:from|due\s*to|after)?\s*(?:medic|taking|treatment|drug|pill|tablet)|nausea\s*(?:from|due|with|after)\s*(?:medic|tak|treatment|drug)|vomit.{0,20}(?:medic|treatment|drug|tablet|pill)|(?<!no\s)side\s*effect(?!s?\s*(?:reported|noted|observed|seen|found))|rash.{0,15}(?:medic|drug|treatment)|hepatotoxic|stomach\s*upset.{0,10}(?:pill|tablet|medic|drug))",
        "label": "Side effects during intensive phase",
    },
    "drtb_long_treatment": {
        "pattern": r"mdr[\-\s]*tb|drug[\-\s]*resist|xdr|dr[\-\s]*tb|second[\-\s]*line|bdq|lzd|bpal|pre[\-\s]*xdr",
        "label": "DR-TB requiring extended treatment (>6 months)",
    },
    "symptom_improvement": {
        "pattern": r"feel(s|ing)?\s*(much\s*)?better|feels?\s*healthy|wants?\s*to\s*stop|asks?\s*(if|about|when).{0,20}stop|stop\s*early|doesn.t\s*need\s*(more\s*)?(medic|treat|pill)|think(s)?\s*(treatment|medicine|he|she).{0,15}(complete|done|over|finish)|can\s*(i|he|she)\s*stop",
        "label": "Symptom improvement at month 2-3",
    },
    "hiv_coinfection": {
        "pattern": r"\bhiv\b\s*(positive|\+|co-?infect|status)|on\s+art\b|antiretroviral|plhiv",
        "label": "HIV co-infection",
    },
}


def augment_factors_from_text(risk_factors: list[dict], user_text: str) -> list[dict]:
    """Scan user input text for factors the model missed.
    
    Only adds a factor if:
    1. The regex matches unambiguously in the source text
    2. The model didn't already extract that factor (no duplicate)
    
    Returns augmented list (original factors + any new ones found).
    """
    if not user_text:
        return risk_factors
    
    text_lower = user_text.lower()
    
    # Determine which ground truth IDs the model already extracted
    already_extracted = set()
    for rf in risk_factors:
        gt = _match_factor_to_ground_truth(rf.get("factor", ""))
        if gt:
            already_extracted.add(gt["factor"])
    
    # Scan text for missing factors
    for factor_id, config in _TEXT_EXTRACTION_PATTERNS.items():
        if factor_id in already_extracted:
            continue
        if re.search(config["pattern"], text_lower):
            # Add the factor
            risk_factors.append({
                "factor": config["label"],
                "visit_first_noted": 1,
                "clinical_basis": "",  # Will be grounded in the next step
                "dots_implication": "Identified from visit notes.",
            })
    
    return risk_factors


# ---------------------------------------------------------------------------
# Main correction pipeline
# ---------------------------------------------------------------------------
def correct_output(output: dict, user_text: str = "") -> dict:
    """Apply all post-processing corrections to model output.
    
    Pipeline:
    1. Ground citations to real evidence
    2. Deduplicate risk factors
    3. Correct tier based on deterministic rules
    
    Returns corrected dict (modifies in place and returns).
    """
    risk_factors = output.get("risk_factors", [])
    treatment_month = output.get("treatment_month", 1)
    
    # 0. Augment: scan user text for factors the model missed
    if user_text:
        risk_factors = augment_factors_from_text(risk_factors, user_text)
    
    # 1. Ground citations
    risk_factors = ground_citations(risk_factors)
    
    # 2. Deduplicate
    risk_factors = deduplicate_factors(risk_factors)
    
    # 3. Tier: Blend deterministic calculation with model prediction.
    # correct_tier() is 100% on ground truth but under-predicts when the model
    # outputs factors that don't match our evidence base (treated as unknown/weak).
    # Strategy:
    #   - All factors matched → use calc_tier (most reliable)
    #   - No factors matched → trust model tier
    #   - Partial match → if they agree, use that; if they disagree by 1 step,
    #     use the higher; if they disagree by 2+ steps, use calc_tier (model is likely wrong)
    model_tier = output.get("dropout_risk_tier", "moderate")
    calc_tier = correct_tier(risk_factors, treatment_month)
    
    # Count how many of the model's factors actually match ground truth
    demographic_ids = {"male_sex", "young_adult"}
    matched_count = 0
    for rf in risk_factors:
        gt = _match_factor_to_ground_truth(rf.get("factor", ""))
        if gt and gt["factor"] not in demographic_ids:
            matched_count += 1
    
    tier_order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    order_tier = {0: "low", 1: "moderate", 2: "high", 3: "critical"}
    
    if len(risk_factors) == 0:
        final_tier = "low"
    elif matched_count == len(risk_factors):
        # All factors matched → fully trust deterministic calc
        final_tier = calc_tier
    elif matched_count == 0:
        # Nothing matched → trust model but cap at "high" (can't verify critical)
        final_tier = model_tier if model_tier != "critical" else "high"
    else:
        # Partial match → reconcile
        model_rank = tier_order.get(model_tier, 1)
        calc_rank = tier_order.get(calc_tier, 1)
        diff = abs(model_rank - calc_rank)
        if diff <= 1:
            # Close enough — but be conservative about upgrading from "low"
            if calc_rank == 0 and model_rank == 1:
                # calc says low, model says moderate — only upgrade if there's
                # a strong matched factor (otherwise thin evidence doesn't justify upgrade)
                has_strong = any(
                    _match_factor_to_ground_truth(rf.get("factor", "")) and
                    ((_match_factor_to_ground_truth(rf.get("factor", ""))["odds_ratio_low"] +
                      _match_factor_to_ground_truth(rf.get("factor", ""))["odds_ratio_high"]) / 2 >= 2.0)
                    and _match_factor_to_ground_truth(rf.get("factor", ""))["factor"] not in demographic_ids
                    for rf in risk_factors
                )
                final_tier = "moderate" if has_strong else "low"
            else:
                final_tier = order_tier[max(model_rank, calc_rank)]
        else:
            # Large disagreement — calc_tier is evidence-based, trust it
            final_tier = calc_tier
    
    # Apply corrections
    output["risk_factors"] = risk_factors
    output["dropout_risk_tier"] = final_tier
    
    # Ensure required fields exist with defaults
    output.setdefault("confidence", "moderate")
    output.setdefault("data_completeness", "incomplete — some fields missing")
    output.setdefault("recommended_action", "Schedule follow-up with DOTS provider")
    
    return output


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with the actual model output from Rakesh Patel
    sample_output = {
        "patient_id": "RP40M_001",
        "visit_number": 3,
        "treatment_month": 3,
        "dropout_risk_tier": "critical",
        "risk_factors": [
            {
                "factor": "Alcohol use",
                "visit_first_noted": 1,
                "clinical_basis": "Alcohol use is associated with increased risk of treatment discontinuation (Smith et al., 2018, OR 3.5)",
                "dots_implication": "Assess alcohol use frequency. Provide counseling and referral for substance use disorder."
            },
            {
                "factor": "Socioeconomic disadvantage (Daily wage painter, BPL card holder)",
                "visit_first_noted": 1,
                "clinical_basis": "Low socioeconomic status and informal employment are associated with increased risk (Jones et al., 2019, OR 1.8)",
                "dots_implication": "Ensure patient has reliable access to food, housing, and transportation."
            },
            {
                "factor": "Substance use (Alcohol)",
                "visit_first_noted": 1,
                "clinical_basis": "Alcohol use is associated with increased risk of treatment discontinuation (Smith et al., 2018, OR 3.5)",
                "dots_implication": "Address alcohol use immediately. Provide supportive counseling."
            },
            {
                "factor": "Physical symptoms during treatment (Nausea)",
                "visit_first_noted": 2,
                "clinical_basis": "Treatment-related adverse events are a common reason for dropout (WHO, 2021)",
                "dots_implication": "Discuss nausea with the patient. Adjust medication timing if necessary."
            },
            {
                "factor": "Financial hardship (Increasing bus fare costs)",
                "visit_first_noted": 3,
                "clinical_basis": "Financial barriers such as transportation costs increase dropout risk (Brown et al., 2020, OR 1.7)",
                "dots_implication": "Explore community DOT or transport assistance options."
            }
        ],
        "recommended_action": "Incorporate brief counseling regarding alcohol use into routine DOTS visits.",
        "data_completeness": "Sputum results and adherence details missing.",
        "confidence": "high"
    }
    
    print("=== BEFORE CORRECTION ===")
    print(f"Tier: {sample_output['dropout_risk_tier']}")
    print(f"Factors ({len(sample_output['risk_factors'])}):")
    for rf in sample_output["risk_factors"]:
        print(f"  - {rf['factor']}")
        print(f"    Citation: {rf['clinical_basis']}")
    
    corrected = correct_output(sample_output)
    
    print("\n=== AFTER CORRECTION ===")
    print(f"Tier: {corrected['dropout_risk_tier']}")
    print(f"Factors ({len(corrected['risk_factors'])}):")
    for rf in corrected["risk_factors"]:
        print(f"  - {rf['factor']}")
        print(f"    Citation: {rf['clinical_basis']}")
    
    print(f"\nFull corrected output:")
    print(json.dumps(corrected, indent=2))
