#!/usr/bin/env python3
"""Generate a synthetic temporal literature graph with *planted* discovery mechanisms.

Mechanisms (each realised after a random publication delay):
  drug -INHIBITS-> gene, gene -ASSOCIATED_WITH-> disease  =>  drug -TREATS-> disease
  gene -INCREASES_EXPRESSION_OF-> gene -INCREASES_EXPRESSION_OF-> gene  =>  first AFFECTS_EXPRESSION_OF third
  gene -ASSOCIATED_WITH-> disease, gene -INCREASES_EXPRESSION_OF-> gene2  =>  gene2 IS_A_BIOMARKER_FOR disease
plus random noise edges. Useful to check that backtest beats the null baseline.

usage: python3 make_toy_data.py > toy_triples.csv
"""
import csv
import random
import sys

rng = random.Random(42)
genes = ["GENE%02d" % i for i in range(1, 81)]
drugs = ["DRUG%02d" % i for i in range(1, 31)]
diseases = ["DISEASE%02d" % i for i in range(1, 21)]
first = {}


def add(s, r, o, y):
    if s == o or y > 2024:
        return False
    k = (s, r, o)
    if k not in first or y < first[k]:
        first[k] = y
        return True
    return False


# base facts
for _ in range(220):
    add(rng.choice(genes), "INCREASES_EXPRESSION_OF", rng.choice(genes), rng.randint(1985, 2020))
for _ in range(120):
    add(rng.choice(genes), "ASSOCIATED_WITH", rng.choice(diseases), rng.randint(1985, 2020))
for _ in range(90):
    add(rng.choice(drugs), "INHIBITS", rng.choice(genes), rng.randint(1985, 2020))
# noise
rels = ["INHIBITS", "ASSOCIATED_WITH", "INCREASES_EXPRESSION_OF", "TREATS"]
for _ in range(150):
    a = rng.choice(genes + drugs)
    b = rng.choice(genes + diseases)
    add(a, rng.choice(rels), b, rng.randint(1985, 2024))

# propagate planted mechanisms (a few rounds so discoveries enable discoveries)
for _round in range(3):
    snap = dict(first)
    by_s = {}
    for (s, r, o), y in snap.items():
        by_s.setdefault((s, r), []).append((o, y))
    for (s, r, o), y in snap.items():
        if r == "INHIBITS":
            for d, y2 in by_s.get((o, "ASSOCIATED_WITH"), []):
                if rng.random() < 0.55:
                    add(s, "TREATS", d, max(y, y2) + rng.randint(1, 8))
        if r == "INCREASES_EXPRESSION_OF":
            for g3, y2 in by_s.get((o, "INCREASES_EXPRESSION_OF"), []):
                if rng.random() < 0.45:
                    add(s, "AFFECTS_EXPRESSION_OF", g3, max(y, y2) + rng.randint(1, 8))
        if r == "ASSOCIATED_WITH":
            for g2, y2 in by_s.get((s, "INCREASES_EXPRESSION_OF"), []):
                if rng.random() < 0.35:
                    add(g2, "IS_A_BIOMARKER_FOR", o, max(y, y2) + rng.randint(1, 8))

w = csv.writer(sys.stdout)
w.writerow(["subject", "relation", "object", "year"])
for (s, r, o), y in sorted(first.items(), key=lambda x: (x[1], x[0])):
    w.writerow([s, r, o, y])
