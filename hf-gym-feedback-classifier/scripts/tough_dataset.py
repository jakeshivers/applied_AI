# scripts/make_harder_dataset.py
"""
Generate a harder synthetic dataset for gym feedback classification.

- Adds lexical overlap between classes
- Inserts typos, slang, negations, hedging
- Varies phrasing/templates so validation isn't trivial
- Creates a true hold-out test set with unseen phrasing

Outputs:
  data/gym_feedback_v2.csv  (train+valid pool)
  data/test.csv             (manual-style unseen phrasing)
"""
from __future__ import annotations
import random, csv, os, re
from pathlib import Path
random.seed(1234)

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["cancel_intent", "billing_issue", "facility_complaint", "positive_feedback", "class_request"]

# Phrase banks with intentional overlap across labels
INTENTS = {
    "cancel_intent": [
        "i'm thinking about cancelling", "please cancel my membership",
        "need to stop the monthly charge", "freeze or cancel my account",
        "moving soon so cancel please", "how do i end my contract",
        "i might have to cancel bc money is tight"
    ],
    "billing_issue": [
        "i was double charged", "why is my monthly fee higher",
        "late fee seems wrong", "charged after i cancelled",
        "refund the extra charge please", "billing is off this month",
        "updated card but still got charged"
    ],
    "facility_complaint": [
        "showers are cold again", "treadmill is broken",
        "locker room is dirty", "no soap in bathrooms",
        "ac not working and too crowded", "smelly mats need cleaning",
        "equipment needs maintenance asap"
    ],
    "positive_feedback": [
        "love the staff and trainers", "great atmosphere and clean",
        "enjoying my membership a lot", "coaches are super helpful",
        "five stars amazing gym", "nice classes and friendly team",
        "everything looks spotless lately"
    ],
    "class_request": [
        "please add more yoga classes", "earlier spin would be great",
        "weekend hiit sessions please", "more bjj classes at night",
        "kettlebell workshops request", "longer sauna hours please",
        "more beginner friendly options"
    ],
}

# Shared tokens/synonyms to create overlap
SHARED_BITS = [
    "membership", "charge", "billing", "refund", "cancel", "freeze",
    "dirty", "clean", "staff", "trainer", "class", "schedule", "peak hours",
    "crowded", "maintenance", "equipment", "sauna", "showers", "card"
]

HEDGES = ["maybe", "sort of", "kinda", "thinking", "might", "possibly", "idk"]
POLITE = ["please", "thanks", "ty", "much appreciated"]
URGENCY = ["asap", "right away", "this week", "before next billing date"]
NEGATIONS = ["not", "never", "no"]

def typo(s: str) -> str:
    # random character drops/swaps on small probability
    if len(s) < 5 or random.random() > 0.3: 
        return s
    i = random.randrange(1, len(s)-1)
    return s[:i] + s[i+1:] + s[i] + s[i+1:]  # simple swap

def sprinkle(words: list[str]) -> str:
    bits = []
    for w in words:
        if random.random() < 0.6:
            bits.append(w)
    return " ".join(bits)

def augment(base: str, label: str) -> str:
    parts = [base]
    if random.random() < 0.6:
        parts.append(random.choice(POLITE))
    if random.random() < 0.5:
        parts.append(random.choice(URGENCY))
    if random.random() < 0.5:
        parts.append(random.choice(HEDGES))
    if random.random() < 0.5:
        parts.append(sprinkle(random.sample(SHARED_BITS, k=random.randint(1,3))))
    if random.random() < 0.4:
        parts.append(random.choice(NEGATIONS) + " sure")
    out = " ".join(parts)
    out = typo(out)
    # lowercase randomly to mimic mobile texting
    if random.random() < 0.5:
        out = out.lower()
    # sometimes join two intent phrases to create ambiguity
    if random.random() < 0.2:
        other_label = random.choice([l for l in LABELS if l != label])
        out += " | " + random.choice(INTENTS[other_label]).lower()
    return re.sub(r"\s+", " ", out).strip()

def generate(n_per_label=250):
    rows = []
    for lab in LABELS:
        for _ in range(n_per_label):
            base = random.choice(INTENTS[lab])
            rows.append({"text": augment(base, lab), "label": lab})
    random.shuffle(rows)
    return rows

def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text","label"])
        w.writeheader()
        w.writerows(rows)

# Train/valid pool
rows = generate(n_per_label=250)     # 1250 rows total
write_csv(OUT_DIR / "gym_feedback_v2.csv", rows)

# True hold-out test with unseen formulations (no augmentation)
TEST_BANK = {
    "cancel_intent": [
        "please terminate my membership", "how do i stop being billed each month",
        "i'd like to cancel before next charge"
    ],
    "billing_issue": [
        "my invoice is wrong", "monthly fee jumped for no reason",
        "charged twice on the same day"
    ],
    "facility_complaint": [
        "bathroom supplies are out", "elliptical squeaks and needs service",
        "no air conditioning on the floor"
    ],
    "positive_feedback": [
        "front desk is awesome", "trainers are fantastic and gym is spotless",
        "really happy with everything lately"
    ],
    "class_request": [
        "could we get more beginner yoga", "any chance for 6am spin",
        "please add weekend kettlebell"
    ],
}
test_rows = [{"text": t, "label": lab} for lab, texts in TEST_BANK.items() for t in texts]
random.shuffle(test_rows)
write_csv(OUT_DIR / "test.csv", test_rows)

print("Wrote:", OUT_DIR / "gym_feedback_v2.csv", "and", OUT_DIR / "test.csv")
