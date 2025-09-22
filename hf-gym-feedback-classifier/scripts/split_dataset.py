
import argparse, csv, random, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            rows.append(row)

    random.shuffle(rows)
    n_valid = int(len(rows) * args.valid_frac)
    valid = rows[:n_valid]
    train = rows[n_valid:]

    for out_path, subset in [(args.train_csv, train), (args.valid_csv, valid)]:
        os.makedirs(Path(out_path).parent, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text","label"])
            w.writeheader()
            w.writerows(subset)

    print(f"Wrote {len(train)} train and {len(valid)} valid rows.")

if __name__ == "__main__":
    main()
