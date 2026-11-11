"""Nested training subsets for the learning curve (research/log.md section 13A).

For seed s the 540 training ids are shuffled with random.Random(13000 + s); the first 270 and the first 135 of that
order are written to configs/splits/dhf1k_train{270,135}_s{s}.txt (sorted, one id per line).

    python scripts/make_lc_splits.py
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs/splits"


def main():
    ids = sorted(int(x) for x in (SPLITS / "dhf1k_train540.txt").read_text().split())
    assert len(ids) == 540 and len(set(ids)) == 540
    hold = {int(x) for x in (SPLITS / "dhf1k_holdout60.txt").read_text().split()}
    for s in range(3):
        order = ids[:]
        random.Random(13000 + s).shuffle(order)
        for n in (270, 135):
            sub = sorted(order[:n])
            assert not hold & set(sub)
            (SPLITS / f"dhf1k_train{n}_s{s}.txt").write_text("\n".join(str(i) for i in sub) + "\n", newline="\n")
            print(f"seed {s}: {n} videos, first ids {sub[:5]}")
        assert set(order[:135]) <= set(order[:270])


if __name__ == "__main__":
    main()
