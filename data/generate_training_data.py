"""
SAHAYAK-TB — Synthetic training data generator.

Generates instruction-tuned training pairs for fine-tuning Gemma 4 E2B
on TB adherence dropout risk assessment.

Two modes:
  1. Rule-based generation (no API needed, unlimited, deterministic)
  2. LLM-augmented generation (uses Gemini API free tier for naturalistic notes)

Usage:
  python generate_training_data.py --mode rule --count 500 --output train.jsonl
  python generate_training_data.py --mode rule --count 100 --output eval.jsonl --seed 9999
  python generate_training_data.py --mode llm --count 200 --output train_llm.jsonl
"""

import argparse
import json
import random
import hashlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Load ground-truth data
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

with open(DATA_DIR / "risk_factors.json") as f:
    RISK_FACTORS: list[dict] = json.load(f)

with open(DATA_DIR / "ntep_actions.json") as f:
    NTEP_ACTIONS: dict = json.load(f)

ACTIONS_BY_ID = {a["id"]: a for a in NTEP_ACTIONS["actions"]}

# ---------------------------------------------------------------------------
# Patient templates
# ---------------------------------------------------------------------------
MALE_NAMES = [
    "Ramesh Kumar", "Suresh Yadav", "Anil Verma", "Rajesh Pandey",
    "Mohammed Iqbal", "Vikram Singh", "Deepak Sharma", "Sanjay Gupta",
    "Ravi Prasad", "Manoj Tiwari", "Sunil Chauhan", "Ajay Pal",
    "Dinesh Rawat", "Harish Joshi", "Kamal Nath", "Bhola Paswan",
    "Mukesh Sahu", "Pappu Yadav", "Guddu Rajbhar", "Firoz Khan",
]

FEMALE_NAMES = [
    "Priya Sharma", "Geeta Bai", "Fatima Begum", "Sunita Devi",
    "Lakshmi Rani", "Meena Kumari", "Anita Gupta", "Pooja Singh",
    "Kavita Yadav", "Rekha Verma", "Shalini Mishra", "Pushpa Devi",
    "Nirmala Bai", "Radha Kumari", "Sarita Pandey", "Shabnam Khatun",
    "Amina Bibi", "Kamla Devi", "Renu Sharma", "Babita Rawat",
]

OCCUPATIONS_MALE = [
    ("daily wage construction worker", True),
    ("auto-rickshaw driver", False),
    ("factory worker", True),
    ("farmer", False),
    ("truck driver", True),
    ("street vendor", True),
    ("painter", True),
    ("security guard", False),
    ("shop worker", False),
    ("migrant laborer", True),
    ("electrician", False),
    ("plumber", True),
]

OCCUPATIONS_FEMALE = [
    ("homemaker", False),
    ("domestic help", True),
    ("ASHA worker", False),
    ("anganwadi worker", False),
    ("agricultural laborer", True),
    ("school teacher", False),
    ("tailor", False),
    ("daily wage worker", True),
    ("vegetable vendor", True),
    ("self-help group member", False),
]

LOCATIONS = [
    ("urban slum", 10, 20),
    ("rural village", 40, 120),
    ("small town", 15, 30),
    ("peri-urban area", 20, 45),
    ("labour colony", 15, 40),
    ("resettlement colony", 10, 25),
    ("tribal area", 60, 180),
]

TREATMENT_CATEGORIES = [
    ("Category I", "new case", 6, False),
    ("Category II", "retreatment case", 8, True),
    ("MDR-TB", "drug-resistant case", 11, True),
    ("MDR-TB shorter", "drug-resistant case on shorter regimen", 9, True),
]

# ---------------------------------------------------------------------------
# Natural language templates — English and Hindi-English mix
# ---------------------------------------------------------------------------
SIDE_EFFECT_PHRASES = {
    "english": [
        "complains of nausea after taking medicine",
        "reports joint pain from medication",
        "skin itching and mild rash",
        "says medicine makes him very drowsy",
        "stomach upset and loss of appetite",
        "dizziness reported after morning dose",
        "vision slightly blurred — needs eye check",
    ],
    "mixed": [
        "ji machlata hai dawai lene ke baad",
        "pet me jalan hoti hai roz",
        "ulti aati hai subah",
        "jodon me dard ho raha hai",
        "dawai se bahut neend aati hai",
        "khana hazam nahi hota",
        "chakkar aate hain dawai ke baad",
    ],
}

FEELING_BETTER_PHRASES = {
    "english": [
        "says he feels much better now",
        "no more cough, feels healthy",
        "thinks treatment might be complete",
        "asks if he can stop early",
        "wants to know how much longer",
        "says she doesn't think she needs medicine anymore",
    ],
    "mixed": [
        "ab theek lag raha hai",
        "khansi band ho gayi hai",
        "kehta hai dawai ki zaroorat nahi",
        "poochh raha hai ki kab band hogi dawai",
        "bolta hai bilkul theek ho gaya",
        "kehti hai ab takleef nahi hai",
    ],
}

TRANSPORT_COST_PHRASES = {
    "english": [
        "says bus fare is getting expensive",
        "transport cost is a burden",
        "can't afford auto fare every day",
        "has to borrow money for travel",
        "spent more on travel than he earns some days",
    ],
    "mixed": [
        "aane me bahut paisa lagta hai",
        "bus ka kiraya mushkil hai",
        "auto ka kharcha nahi utha sakta",
        "paisa nahi hai aane ka",
        "udhar lena padta hai aane ke liye",
    ],
}

WORK_PRESSURE_PHRASES = {
    "english": [
        "can't afford to miss work for clinic visits",
        "employer warned about absences",
        "lost wages this week for coming to DOT center",
        "work pressure is increasing",
        "says boss will fire him if he takes more leave",
    ],
    "mixed": [
        "kaam chhod ke nahi aa sakta",
        "mazdoori nahi milegi agar nahi gaya",
        "seth ne daanta ki bahut chutti le raha hai",
        "paise ki bahut tangi hai",
        "rojana ki kamai chhootjati hai",
    ],
}

ALCOHOL_PHRASES = {
    "english": [
        "drinks daily", "alcohol use reported", "family says he drinks heavily",
        "admits to regular drinking", "still drinking despite advice",
    ],
    "mixed": [
        "sharab peeta hai daily", "daaru pi ke aata hai",
        "ghar wale kehte hain bahut peeta hai", "sharab nahi chhodi",
        "roz peeta hai thoda thoda",
    ],
}

SMOKING_PHRASES = {
    "english": [
        "smokes regularly", "still smoking despite advice", "chews tobacco",
        "bidi smoker", "gutka user",
    ],
    "mixed": [
        "bidi peeta hai", "tambaaku khata hai", "gutka ki aadat hai",
        "bidi nahi chhodi", "subah se shaam bidi peeta hai",
    ],
}

MIGRATION_PHRASES = {
    "english": [
        "migrant worker, no permanent address here",
        "plans to move to another city for work",
        "temporary stay at construction site",
        "may go back to village for a month",
        "shifts between cities for work",
    ],
    "mixed": [
        "pravasi mazdoor hai, yahan temporary hai",
        "gaon jaana hai next month",
        "doosre sheher jaana padega kaam ke liye",
        "yahan pakka address nahi hai",
        "kaam ke liye idhar udhar jaata rehta hai",
    ],
}

STIGMA_PHRASES = {
    "english": [
        "hasn't told employer about TB",
        "afraid of being found out",
        "neighbors don't know",
        "hides medicine from colleagues",
    ],
    "mixed": [
        "kisi ko nahi bataya bimari ke baare me",
        "sharam aati hai",
        "darr hai ki kaam chala jayega",
        "padosiyon ko nahi pata",
    ],
}

MISSED_DOSE_PHRASES = {
    "english": [
        "missed 2 days last week",
        "skipped a few doses",
        "came late by 3 days",
        "missed entire last week",
        "ASHA worker found him — hadn't come in 10 days",
    ],
    "mixed": [
        "do din nahi aaya tha",
        "kuch dose miss ho gaye",
        "teen din late aaya",
        "poora hafta nahi aaya",
        "ASHA didi ne dhundh ke laaya — 10 din se nahi aaya tha",
    ],
}


def _pick_lang(lang_preference: str) -> str:
    if lang_preference == "english":
        return "english"
    elif lang_preference == "hindi":
        return "mixed"
    else:
        return random.choice(["english", "mixed"])


def _pick_phrase(phrase_dict: dict, lang: str) -> str:
    key = lang if lang in phrase_dict else "english"
    return random.choice(phrase_dict[key])


# ---------------------------------------------------------------------------
# Case generator
# ---------------------------------------------------------------------------
def generate_case(rng: random.Random) -> dict:
    """Generate a single synthetic case with visit notes and risk assessment."""
    # Demographics
    is_male = rng.random() < 0.6  # TB prevalence higher in males
    if is_male:
        name = rng.choice(MALE_NAMES)
        age = rng.randint(18, 65)
        occupation, is_daily_wage = rng.choice(OCCUPATIONS_MALE)
        pronoun = "he"
    else:
        name = rng.choice(FEMALE_NAMES)
        age = rng.randint(18, 60)
        occupation, is_daily_wage = rng.choice(OCCUPATIONS_FEMALE)
        pronoun = "she"

    sex = "M" if is_male else "F"
    location_name, travel_min_low, travel_min_high = rng.choice(LOCATIONS)
    travel_time = rng.randint(travel_min_low, travel_min_high)
    weight_start = rng.randint(38, 65)

    # Treatment — skew toward Category I (most common in reality)
    cat_weights = [60, 20, 12, 8]  # Cat I, Cat II, MDR, MDR shorter
    cat_label, cat_desc, total_months, is_complex = rng.choices(TREATMENT_CATEGORIES, weights=cat_weights, k=1)[0]

    # How many visits and what month range
    num_visits = rng.randint(2, 4)
    visit_months = sorted(rng.sample(range(1, min(total_months, 7) + 1), min(num_visits, 6)))
    if not visit_months:
        visit_months = [1, 2]
    num_visits = len(visit_months)

    # Language mix
    lang_preference = rng.choices(["english", "mixed", "hindi"], weights=[50, 35, 15])[0]
    lang = _pick_lang(lang_preference)

    # Select risk factors for this patient
    num_risk_factors = _sample_risk_count(rng)
    available_factors = list(RISK_FACTORS)

    # Conditional: if not male, remove male_sex factor from pool
    if not is_male:
        available_factors = [f for f in available_factors if f["factor"] != "male_sex"]
    # If not retreatment, remove retreatment factor
    if cat_label == "Category I":
        available_factors = [f for f in available_factors if f["factor"] != "retreatment_case"]
    else:
        # Retreatment — ensure retreatment factor is included
        retreatment_f = [f for f in RISK_FACTORS if f["factor"] == "retreatment_case"]
        if retreatment_f and retreatment_f[0] not in available_factors[:num_risk_factors]:
            pass  # will be handled below
    if "MDR" not in cat_label:
        available_factors = [f for f in available_factors if f["factor"] != "drtb_long_treatment"]

    # Weight factors by OR for more realistic sampling
    weights = [(f["odds_ratio_low"] + f["odds_ratio_high"]) / 2 for f in available_factors]
    selected_factors = []
    if num_risk_factors > 0 and available_factors:
        selected_factors = _weighted_sample(rng, available_factors, weights, min(num_risk_factors, len(available_factors)))

    # Force certain factors based on demographics — but only sometimes
    factor_ids = {f["factor"] for f in selected_factors}
    if is_daily_wage and "daily_wage_poverty" not in factor_ids:
        daily_f = [f for f in RISK_FACTORS if f["factor"] == "daily_wage_poverty"]
        if daily_f and rng.random() < 0.3:
            selected_factors.append(daily_f[0])
    if travel_time > 90 and "distance_from_dot_center" not in factor_ids:
        dist_f = [f for f in RISK_FACTORS if f["factor"] == "distance_from_dot_center"]
        if dist_f and rng.random() < 0.4:
            selected_factors.append(dist_f[0])
    if cat_label == "Category II" and "retreatment_case" not in factor_ids:
        rt_f = [f for f in RISK_FACTORS if f["factor"] == "retreatment_case"]
        if rt_f:
            selected_factors.append(rt_f[0])
    if "MDR" in cat_label and "drtb_long_treatment" not in factor_ids:
        dr_f = [f for f in RISK_FACTORS if f["factor"] == "drtb_long_treatment"]
        if dr_f:
            selected_factors.append(dr_f[0])

    # Determine risk tier
    risk_tier = compute_risk_tier(selected_factors, max(visit_months))

    # Fix 4: Temporal pattern — decide when each factor first appears
    # Some factors are present from Visit 1 (demographics, comorbidities)
    # Others EMERGE during treatment (side effects, feeling better, migration plans)
    factor_visit_map = {}  # factor_id -> visit_index where it first appears
    early_factors = {"alcohol_use", "smoking_tobacco", "daily_wage_poverty", "male_sex",
                     "young_adult", "hiv_coinfection", "diabetes_comorbidity",
                     "divorced_separated", "low_health_literacy", "distance_from_dot_center",
                     "retreatment_case", "drtb_long_treatment"}
    late_factors = {"symptom_improvement", "intensive_phase_side_effects",
                    "migration_no_fixed_address"}

    for f in selected_factors:
        fid = f["factor"]
        if fid in early_factors:
            factor_visit_map[fid] = 0  # Always present from visit 1
        elif fid in late_factors:
            # Emerges at visit 2+ (or visit 1 if only 1 visit)
            factor_visit_map[fid] = min(1, num_visits - 1)
        else:
            # Random: 60% from visit 1, 40% emerges later
            if rng.random() < 0.6 or num_visits <= 1:
                factor_visit_map[fid] = 0
            else:
                factor_visit_map[fid] = rng.randint(1, num_visits - 1)

    # Data completeness — randomly omit some fields
    missing_fields = []
    if rng.random() < 0.3:
        missing_fields.append(rng.choice([
            "weight not recorded at visit {}".format(rng.choice(visit_months[1:])) if len(visit_months) > 1 else "weight not recorded",
            "sputum conversion status not confirmed",
            "side effect severity not quantified",
            "exact medication timing not recorded",
            "family support status not assessed",
        ]))

    # Generate visit notes
    visit_notes = _generate_visit_notes(
        rng, name, age, sex, occupation, location_name, travel_time,
        weight_start, cat_label, cat_desc, visit_months, selected_factors,
        lang, total_months, factor_visit_map
    )

    # Generate output
    patient_id = _hash_id(name, age, sex)
    output = _build_output(
        patient_id, num_visits, max(visit_months), risk_tier,
        selected_factors, visit_months, missing_fields, factor_visit_map
    )

    return {
        "input": visit_notes,
        "output": output,
        "metadata": {
            "risk_tier": risk_tier,
            "num_visits": num_visits,
            "language": lang_preference,
            "num_factors": len(selected_factors),
            "treatment_category": cat_label,
            "has_temporal_emergence": any(v > 0 for v in factor_visit_map.values()),
        }
    }


def _sample_risk_count(rng: random.Random) -> int:
    """Sample number of risk factors.
    
    Target tier distribution after compute_risk_tier: ~20/30/30/20 (low/mod/high/crit).
    Forced factors (retreatment, MDR-TB, daily wage, distance) add to this count,
    so we sample conservatively.
    """
    r = rng.random()
    if r < 0.30:
        return 0  # likely low risk
    elif r < 0.60:
        return 1  # likely moderate
    elif r < 0.80:
        return 2  # likely high
    else:
        return rng.randint(3, 5)  # likely critical


def _weighted_sample(rng: random.Random, items: list, weights: list, k: int) -> list:
    """Weighted sampling without replacement."""
    items = list(items)
    weights = list(weights)
    selected = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        if total <= 0:
            break
        r = rng.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                selected.append(items.pop(i))
                weights.pop(i)
                break
    return selected


def compute_risk_tier(factors: list[dict], treatment_month: int) -> str:
    """Compute risk tier from selected factors and treatment month.
    
    Calibration notes:
    - Demographic-only factors (male_sex, young_adult) should not alone trigger high.
    - 'high' requires multiple behavioral/clinical factors with strong evidence.
    - 'critical' requires compounding factors over time.
    """
    if not factors:
        return "low"

    # Separate demographic factors from behavioral/clinical ones
    demographic_only = {"male_sex", "young_adult"}
    clinical_factors = [f for f in factors if f["factor"] not in demographic_only]
    all_factors = factors

    # Use mean OR (not high end) for threshold decisions
    def mean_or(f):
        return (f["odds_ratio_low"] + f["odds_ratio_high"]) / 2

    strong_clinical = [f for f in clinical_factors if mean_or(f) >= 2.0]
    moderate_clinical = [f for f in clinical_factors if 1.5 <= mean_or(f) < 2.0]

    # Critical: 3+ strong clinical factors AND treatment month > 3
    if len(strong_clinical) >= 3 and treatment_month > 3:
        return "critical"
    # Critical: 2+ strong clinical + 1+ moderate AND month > 3
    if len(strong_clinical) >= 2 and len(moderate_clinical) >= 1 and treatment_month > 3:
        return "critical"
    # High: 2+ strong clinical factors
    if len(strong_clinical) >= 2:
        return "high"
    # High: 1 strong clinical + 1+ moderate clinical factor
    if len(strong_clinical) >= 1 and len(moderate_clinical) >= 1:
        return "high"
    # High: 1 strong clinical + 2+ total factors (including demographic)
    if len(strong_clinical) >= 1 and len(all_factors) >= 3:
        return "high"
    # High: any single clinical factor with mean OR > 2.5
    if any(mean_or(f) > 2.5 for f in clinical_factors):
        return "high"
    # Moderate: 1 strong clinical factor (mean OR 2.0-2.5)
    if len(strong_clinical) >= 1:
        return "moderate"
    # Moderate: 2+ moderate clinical factors
    if len(moderate_clinical) >= 2:
        return "moderate"
    # Moderate: 2+ any factors with decent average OR
    if len(all_factors) >= 2:
        avg = sum(mean_or(f) for f in all_factors) / len(all_factors)
        if avg >= 1.5:
            return "moderate"
    # Low: 0-1 factors with low odds ratios
    return "low"


def _hash_id(name: str, age: int, sex: str) -> str:
    initials = "".join(w[0] for w in name.split())
    return f"{initials}_{age}{sex}_{hashlib.md5(name.encode()).hexdigest()[:4]}"


def _generate_visit_notes(
    rng: random.Random, name: str, age: int, sex: str,
    occupation: str, location: str, travel_time: int,
    weight_start: int, cat_label: str, cat_desc: str,
    visit_months: list[int], factors: list[dict],
    lang: str, total_months: int, factor_visit_map: dict = None
) -> str:
    """Generate multi-visit natural language notes."""
    factor_ids = {f["factor"] for f in factors}
    if factor_visit_map is None:
        factor_visit_map = {f["factor"]: 0 for f in factors}
    notes = []
    weight = weight_start
    pronoun = "he" if sex == "M" else "she"
    pronoun_cap = pronoun.capitalize()

    for visit_idx, month in enumerate(visit_months):
        visit_num = visit_idx + 1
        lines = [f"Visit {visit_num} (Month {month}):"]

        # First visit — demographics
        if visit_idx == 0:
            lines.append(f"{name}, {age}{sex}, {occupation}.")
            lines.append(f"{cat_label} {cat_desc}.")
            lines.append(f"Weight {weight}kg. Sputum positive.")
            if travel_time > 30:
                lines.append(f"Lives in {location}, about {travel_time} min from DOT center.")
            else:
                lines.append(f"Lives near DOT center in {location}.")

            # First-visit factor mentions (only factors that appear at visit 0)
            for f in factors:
                fid = f["factor"]
                if factor_visit_map.get(fid, 0) != 0:
                    continue  # This factor emerges later
                if fid == "alcohol_use":
                    lines.append(_pick_phrase(ALCOHOL_PHRASES, lang) + ".")
                elif fid == "smoking_tobacco":
                    lines.append(_pick_phrase(SMOKING_PHRASES, lang) + ".")
                elif fid == "migration_no_fixed_address":
                    lines.append(_pick_phrase(MIGRATION_PHRASES, lang) + ".")
                elif fid == "hiv_coinfection":
                    lines.append("HIV positive, on ART.")
                elif fid == "diabetes_comorbidity":
                    lines.append("Diabetic, on metformin." if lang == "english" else "Sugar bhi hai, metformin le raha hai.")
                elif fid == "divorced_separated":
                    lines.append("Separated from spouse, lives alone." if lang == "english" else "Talaq ho gaya, akela rehta hai.")
                elif fid == "low_health_literacy":
                    lines.append("Illiterate — explained treatment using pictures." if lang == "english" else "Padha likha nahi hai — tasveer se samjhaya.")
                elif fid == "young_adult" and age <= 30:
                    pass  # Age already captured
                elif fid == "daily_wage_poverty":
                    if lang == "english":
                        lines.append(f"Daily wage earner, worried about missing work for treatment.")
                    else:
                        lines.append(f"Rojana ki kamai se ghar chalta hai, kaam ka tension hai.")

            # Fix 1: Explicit NEGATIVE evidence for low-risk cases
            # Teach model that absence of risk + presence of protective factors = low
            if len(factors) == 0:
                neg_evidence_english = [
                    "No alcohol, no tobacco use.",
                    "Family supportive — wife/husband accompanies.",
                    "Patient clearly understands 6-month treatment requirement.",
                    "No financial concerns reported — salaried job.",
                    "Motivated to complete treatment.",
                    "No comorbidities.",
                ]
                neg_evidence_mixed = [
                    "Sharab nahi peeta, tambaaku nahi.",
                    "Parivar ka poora sahyog hai.",
                    "6 mahine ki dawai samajh gaya hai.",
                    "Naukri hai, paison ki dikkat nahi.",
                    "Dawai poori karni hai, samajh gaya.",
                    "Koi aur bimari nahi hai.",
                ]
                evidence = neg_evidence_english if lang == "english" else neg_evidence_mixed
                # Add 2-3 explicit protective statements at visit 1
                n_neg = rng.randint(2, 3)
                for phrase in rng.sample(evidence, min(n_neg, len(evidence))):
                    lines.append(phrase)
            elif len(factors) == 1:
                # Single factor — still mention some protective context
                if rng.random() < 0.5:
                    if lang == "english":
                        lines.append(rng.choice([
                            "Otherwise no other risk factors identified.",
                            "Family is supportive.",
                            "No substance use.",
                        ]))
                    else:
                        lines.append(rng.choice([
                            "Baaki sab theek hai.",
                            "Ghar wale saath hain.",
                            "Nasha nahi karta.",
                        ]))

        # Subsequent visits
        else:
            # Weight change
            if "alcohol_use" in factor_ids or len(factors) > 3:
                weight -= rng.randint(0, 2)
            elif len(factors) <= 1:
                weight += rng.randint(0, 2)
            else:
                weight += rng.choice([-1, 0, 0, 1])
            lines.append(f"Weight {weight}kg.")

            # Protective/adherence phrases for LOW risk cases
            if len(factors) == 0:
                protective_phrases = [
                    "Coming regularly for DOT.",
                    "No complaints. Adherence excellent.",
                    "Taking all medicines on time.",
                    "No side effects reported.",
                    "Family support continues.",
                    "Patient motivated and compliant.",
                    "No missed doses.",
                    "Sputum conversion confirmed." if month >= 2 else "Coming regularly.",
                ]
                # Add 1-2 protective phrases per visit
                n_protective = rng.randint(1, 2)
                chosen = rng.sample(protective_phrases, min(n_protective, len(protective_phrases)))
                for phrase in chosen:
                    lines.append(phrase)
            elif len(factors) == 1 and rng.random() < 0.4:
                # Single factor — still show some adherence
                mild_phrases = [
                    "Otherwise coming regularly.",
                    "No other complaints.",
                    "Taking medicines on time.",
                ]
                lines.append(rng.choice(mild_phrases))

            # Side effects
            if "intensive_phase_side_effects" in factor_ids and month <= 3:
                if factor_visit_map.get("intensive_phase_side_effects", 0) <= visit_idx:
                    lines.append(_pick_phrase(SIDE_EFFECT_PHRASES, lang) + ".")

            # Feeling better (usually month 2-3)
            if "symptom_improvement" in factor_ids and month >= 2:
                if factor_visit_map.get("symptom_improvement", 0) <= visit_idx:
                    lines.append(_pick_phrase(FEELING_BETTER_PHRASES, lang) + ".")

            # Transport / distance
            if "distance_from_dot_center" in factor_ids and rng.random() < 0.6:
                if factor_visit_map.get("distance_from_dot_center", 0) <= visit_idx:
                    lines.append(_pick_phrase(TRANSPORT_COST_PHRASES, lang) + ".")

            # Work pressure
            if "daily_wage_poverty" in factor_ids and rng.random() < 0.5:
                if factor_visit_map.get("daily_wage_poverty", 0) <= visit_idx:
                    lines.append(_pick_phrase(WORK_PRESSURE_PHRASES, lang) + ".")

            # Missed doses
            if len(factors) >= 2 and month >= 2 and rng.random() < 0.4:
                lines.append(_pick_phrase(MISSED_DOSE_PHRASES, lang) + ".")

            # Migration plans — may EMERGE at later visits
            if "migration_no_fixed_address" in factor_ids and month >= 2:
                if factor_visit_map.get("migration_no_fixed_address", 0) <= visit_idx and rng.random() < 0.5:
                    lines.append(_pick_phrase(MIGRATION_PHRASES, lang) + ".")

            # Alcohol continuing
            if "alcohol_use" in factor_ids and rng.random() < 0.6:
                if factor_visit_map.get("alcohol_use", 0) <= visit_idx:
                    lines.append(f"Still drinking." if lang == "english" else "Sharab nahi chhodi.")

            # Smoking continuing
            if "smoking_tobacco" in factor_ids and rng.random() < 0.5:
                if factor_visit_map.get("smoking_tobacco", 0) <= visit_idx:
                    lines.append(f"Still smoking." if lang == "english" else "Bidi nahi chhodi.")

            # NEW factors emerging at this visit — add introduction language
            for f in factors:
                fid = f["factor"]
                if factor_visit_map.get(fid, 0) == visit_idx and visit_idx > 0:
                    # Factor is NEW at this visit — announce it
                    if fid == "alcohol_use":
                        lines.append("New concern: " + _pick_phrase(ALCOHOL_PHRASES, lang) + ".")
                    elif fid == "smoking_tobacco":
                        lines.append("Now reports: " + _pick_phrase(SMOKING_PHRASES, lang) + ".")
                    elif fid == "migration_no_fixed_address":
                        lines.append("New development: " + _pick_phrase(MIGRATION_PHRASES, lang) + ".")
                    elif fid == "daily_wage_poverty":
                        if lang == "english":
                            lines.append("Now reports financial pressure from treatment visits.")
                        else:
                            lines.append("Ab paison ki dikkat bata raha hai.")
                    elif fid == "divorced_separated":
                        if lang == "english":
                            lines.append("Mentions spouse left recently — now lives alone.")
                        else:
                            lines.append("Bataya ki ghar me akela hai ab.")
                    elif fid == "low_health_literacy":
                        if lang == "english":
                            lines.append("Seems confused about treatment duration — re-explained.")
                        else:
                            lines.append("Dawai ke baare me confuse hai — dobara samjhaya.")

            # Stigma
            if rng.random() < 0.15:
                lines.append(_pick_phrase(STIGMA_PHRASES, lang) + ".")

        notes.append(" ".join(lines))

    return "\n\n".join(notes)


def _build_output(
    patient_id: str, num_visits: int, treatment_month: int,
    risk_tier: str, factors: list[dict], visit_months: list[int],
    missing_fields: list[str], factor_visit_map: dict = None
) -> dict:
    """Build the structured risk assessment output."""
    if factor_visit_map is None:
        factor_visit_map = {f["factor"]: 0 for f in factors}

    rng = random.Random(hash(patient_id))  # Deterministic per patient

    risk_factor_entries = []
    for f in factors:
        or_low = f["odds_ratio_low"]
        or_high = f["odds_ratio_high"]
        sources = f["sources"]
        source = sources[0] if sources else "published Indian cohort literature"
        source_alt = sources[1] if len(sources) > 1 else source

        # Fix 4: visit_first_noted from factor_visit_map
        visit_noted = factor_visit_map.get(f["factor"], 0) + 1  # 1-indexed

        # Fix 2: Vary clinical_basis format across samples
        basis_templates = [
            f"{f['label']} — OR {or_low}-{or_high} ({source})",
            f"{f['label']} shows {or_low}-{or_high}x higher dropout odds ({source})",
            f"Evidence: {or_low}-{or_high}x dropout risk associated with {f['label'].lower()} ({source_alt})",
            f"{f['label'].lower()} is a validated predictor of TB treatment dropout with OR {or_low}-{or_high} per {source}",
            f"Indian cohort data shows {or_low}-{or_high}x increased dropout odds when {f['label'].lower()} present ({source})",
            f"OR {or_low}-{or_high} for dropout ({source}; {source_alt})" if source != source_alt else f"OR {or_low}-{or_high} for dropout ({source})",
        ]
        clinical_basis = rng.choice(basis_templates)

        # Get the corresponding NTEP action
        action_id = f.get("ntep_action", "")
        action_detail = ACTIONS_BY_ID.get(action_id, {})
        dots_implication = action_detail.get("description", f"address {f['label']} per NTEP protocol")

        risk_factor_entries.append({
            "factor": f["label"],
            "visit_first_noted": visit_noted,
            "clinical_basis": clinical_basis,
            "dots_implication": dots_implication
        })

    # Fix 5: Composite NTEP action recommendations
    if risk_tier == "critical":
        recommended = "URGENT: Escalate to STS/STLS supervisor and District TB Officer. "
        if len(factors) >= 2:
            # Composite: address top 2 factors
            actions = []
            for f in factors[:2]:
                aid = f.get("ntep_action", "")
                a = ACTIONS_BY_ID.get(aid, {})
                if a:
                    actions.append(a.get("label", ""))
            if actions:
                recommended += "Priority actions: " + "; ".join(actions) + ". "
            recommended += "Schedule home visit within 48 hours if patient misses next dose."
        elif factors:
            top_action = ACTIONS_BY_ID.get(factors[0].get("ntep_action", ""), {})
            recommended += top_action.get("description", "Immediate intervention required.")
    elif risk_tier == "high":
        recommended = "Schedule home visit before next scheduled dose. "
        if len(factors) >= 2:
            # Composite: combine actions
            action_labels = []
            for f in factors[:3]:
                aid = f.get("ntep_action", "")
                a = ACTIONS_BY_ID.get(aid, {})
                if a and a.get("label"):
                    action_labels.append(a["label"])
            if action_labels:
                recommended += "Combined intervention: " + " + ".join(action_labels[:2]) + ". "
            recommended += "Do not wait for missed dose — proactive outreach required."
        elif factors:
            top_action = ACTIONS_BY_ID.get(factors[0].get("ntep_action", ""), {})
            recommended += top_action.get("description", "Do not wait for missed dose.")
    elif risk_tier == "moderate":
        if factors:
            top_action_id = factors[0].get("ntep_action", "")
            top_action = ACTIONS_BY_ID.get(top_action_id, {})
            recommended = top_action.get("description", "Monitor closely and address identified risk factors.")
            if len(factors) >= 2:
                second_action_id = factors[1].get("ntep_action", "")
                second_action = ACTIONS_BY_ID.get(second_action_id, {})
                if second_action:
                    recommended += f" Additionally: {second_action.get('label', '')}."
        else:
            recommended = "Monitor closely and address identified risk factors."
    else:
        recommended = "Continue routine DOT schedule. Acknowledge patient adherence and reinforce completion message."

    # Data completeness
    if missing_fields:
        completeness = "partial — " + "; ".join(missing_fields)
    elif len(factors) == 0:
        completeness = "complete — all key fields recorded across visits"
    else:
        completeness = "adequate — key fields recorded"

    # Confidence
    if missing_fields and len(factors) >= 2:
        confidence = "moderate"
    elif risk_tier in ("high", "critical") and not missing_fields:
        confidence = "high"
    elif risk_tier == "low":
        confidence = "high"
    else:
        confidence = "moderate"

    return {
        "patient_id": patient_id,
        "visit_number": num_visits,
        "treatment_month": treatment_month,
        "dropout_risk_tier": risk_tier,
        "risk_factors": risk_factor_entries,
        "recommended_action": recommended,
        "data_completeness": completeness,
        "confidence": confidence
    }


# ---------------------------------------------------------------------------
# Format for Gemma 4 chat template
# ---------------------------------------------------------------------------
# Fix 6: Previous assessment tiers for multi-turn simulation
_PREV_TIER_TRANSITIONS = {
    "low": ["low"],        # Low stays low or wasn't assessed before
    "moderate": ["low", "moderate"],  # Could have been low before
    "high": ["moderate", "low"],      # Escalated from lower tier
    "critical": ["moderate", "high"], # Escalated from moderate/high
}

def format_for_training(case: dict, system_prompt: str) -> dict:
    """Format a case into Gemma 4 chat template for SFT training."""
    user_content = case["input"]

    # Fix 6: ~20% of cases include a previous assessment context
    # This teaches the model to understand longitudinal progression
    rng = random.Random(hash(user_content))
    if rng.random() < 0.20 and case["metadata"]["num_visits"] >= 3:
        current_tier = case["metadata"]["risk_tier"]
        prev_options = _PREV_TIER_TRANSITIONS.get(current_tier, ["low"])
        prev_tier = rng.choice(prev_options)
        prev_month = max(1, case["output"]["treatment_month"] - 2)

        prev_context = f"[Previous assessment at Month {prev_month}: dropout_risk_tier={prev_tier}]\n\n"
        user_content = prev_context + user_content

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(case["output"], indent=2)}
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SAHAYAK-TB training data")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=str, default="train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = DATA_DIR / args.output

    # Import system prompt from schema
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
    from schema import SYSTEM_PROMPT  # noqa: E402

    cases = []
    tier_counts = {"low": 0, "moderate": 0, "high": 0, "critical": 0}

    print(f"Generating {args.count} cases (seed={args.seed})...")

    for i in range(args.count):
        case = generate_case(rng)
        tier = case["metadata"]["risk_tier"]
        tier_counts[tier] += 1
        formatted = format_for_training(case, SYSTEM_PROMPT)
        cases.append(formatted)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.count}")

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(cases)} cases to {output_path}")
    print(f"Tier distribution: {tier_counts}")
    total = sum(tier_counts.values())
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()
