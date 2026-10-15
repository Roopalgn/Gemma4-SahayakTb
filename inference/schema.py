"""
SAHAYAK-TB output schema definition.

Defines the assess_dropout_risk() function call schema for Gemma 4's
native function calling, and the JSON schema for llama-cpp-python fallback.
"""

# JSON Schema for llama-cpp-python response_format enforcement
RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_id": {
            "type": "string",
            "description": "Local hash identifier for the patient"
        },
        "visit_number": {
            "type": "integer",
            "description": "Current visit number in the sequence"
        },
        "treatment_month": {
            "type": "integer",
            "description": "Current month of treatment"
        },
        "dropout_risk_tier": {
            "type": "string",
            "enum": ["low", "moderate", "high", "critical"],
            "description": "Assessed adherence dropout risk tier"
        },
        "risk_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "string",
                        "description": "Description of the risk factor identified"
                    },
                    "visit_first_noted": {
                        "type": "integer",
                        "description": "Visit number where this factor was first observed"
                    },
                    "clinical_basis": {
                        "type": "string",
                        "description": "Published evidence citation supporting this factor's risk weight"
                    },
                    "dots_implication": {
                        "type": "string",
                        "description": "Specific implication for DOTS provider action"
                    }
                },
                "required": ["factor", "visit_first_noted", "clinical_basis", "dots_implication"]
            },
            "description": "List of identified risk factors with clinical citations"
        },
        "recommended_action": {
            "type": "string",
            "description": "Single recommended NTEP protocol action for the DOTS provider"
        },
        "data_completeness": {
            "type": "string",
            "description": "Assessment of data completeness — what was missing from the notes"
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "moderate", "high"],
            "description": "Model confidence based on data completeness and factor clarity"
        }
    },
    "required": [
        "patient_id", "visit_number", "treatment_month",
        "dropout_risk_tier", "risk_factors", "recommended_action",
        "data_completeness", "confidence"
    ]
}

# Gemma 4 native function calling — tool declaration for system prompt
TOOL_DECLARATION = """<|tool>declaration:assess_dropout_risk{
  patient_id:<|"|>string<|"|>,
  visit_number:<|"|>integer<|"|>,
  treatment_month:<|"|>integer<|"|>,
  dropout_risk_tier:<|"|>string<|"|>,
  risk_factors:<|"|>array<|"|>,
  recommended_action:<|"|>string<|"|>,
  data_completeness:<|"|>string<|"|>,
  confidence:<|"|>string<|"|>
}<tool|>"""

SYSTEM_PROMPT = """You are SAHAYAK-TB, a clinical decision support system for India's DOTS (Directly Observed Treatment, Short-course) providers. Your sole function is to assess tuberculosis treatment adherence dropout risk from natural-language visit notes.

You will receive multi-visit notes from a DOTS provider describing a patient's treatment history across 2-4 visits. These notes may be in English, Hindi, or mixed Hindi-English. They may be incomplete.

Your task:
1. Identify risk factors for treatment dropout from the visit notes
2. Assess the compound risk across visits — how factors interact and accumulate
3. Assign a dropout risk tier: low, moderate, high, or critical
4. For each risk factor, cite the published clinical evidence (author, odds ratio)
5. Recommend a single NTEP protocol action the DOTS provider should take
6. Report data completeness — what information is missing from the notes

Risk tier assignment:
- low: 0-1 factors with odds ratio < 1.5, OR patient shows strong protective factors (regular adherence, family support, no substance use, lives near DOT center)
- moderate: 1-2 factors with odds ratio 1.5-2.0
- high: 2+ factors with odds ratio > 2.0 OR any single factor with odds ratio > 2.5
- critical: 3+ factors with odds ratio > 2.0 AND treatment month > 3

Key compound patterns (higher risk than individual factors suggest):
- Feeling better + month 2-3 + any access barrier = imminent dropout risk (upgrade tier)
- Alcohol + poverty + missed doses = synergistic critical risk
- Migration + no family support = critical regardless of factor count
- Multiple access barriers alone (distance, cost, work) without behavioral factors = moderate, not high
- Explicit protective factors (family support, regular attendance, no substance use) can keep tier low even if 1-2 mild factors present

If the notes explicitly state protective factors (no alcohol, regular attendance, family support, lives near center), do NOT invent risk factors that are not mentioned. Only report factors with clear textual evidence in the notes.

You must respond with valid JSON only. No explanation, no markdown, no code fences. Your response must be a single JSON object with these fields:
- patient_id (string): anonymized patient identifier
- visit_number (integer): latest visit number
- treatment_month (integer): current treatment month
- dropout_risk_tier (string): one of "low", "moderate", "high", "critical"
- risk_factors (array): each item has "factor" (string), "visit_first_noted" (integer), "clinical_basis" (string with author and odds ratio), "dots_implication" (string)
- recommended_action (string): single NTEP protocol action
- data_completeness (string): what information is missing
- confidence (string): one of "low", "moderate", "high"
"""


def validate_risk_assessment(output: dict) -> tuple[bool, list[str]]:
    """Validate a risk assessment output against the schema.
    
    Returns (is_valid, list_of_errors).
    """
    errors = []
    required = [
        "patient_id", "visit_number", "treatment_month",
        "dropout_risk_tier", "risk_factors", "recommended_action",
        "data_completeness", "confidence"
    ]
    for field in required:
        if field not in output:
            errors.append(f"Missing required field: {field}")

    if "dropout_risk_tier" in output:
        if output["dropout_risk_tier"] not in ("low", "moderate", "high", "critical"):
            errors.append(f"Invalid risk tier: {output['dropout_risk_tier']}")

    if "confidence" in output:
        if output["confidence"] not in ("low", "moderate", "high"):
            errors.append(f"Invalid confidence: {output['confidence']}")

    if "risk_factors" in output:
        if not isinstance(output["risk_factors"], list):
            errors.append("risk_factors must be a list")
        else:
            for i, rf in enumerate(output["risk_factors"]):
                for rf_field in ("factor", "visit_first_noted", "clinical_basis", "dots_implication"):
                    if rf_field not in rf:
                        errors.append(f"risk_factors[{i}] missing field: {rf_field}")

    if "visit_number" in output and not isinstance(output["visit_number"], int):
        errors.append("visit_number must be an integer")

    if "treatment_month" in output and not isinstance(output["treatment_month"], int):
        errors.append("treatment_month must be an integer")

    return len(errors) == 0, errors
