# Evaluation: back-testing knowledge prediction

## Protocol (paper §4.2–4.3, App. C)
1. Choose cutoff `t_k`. Train/score only with triples first observed ≤ `t_k`.
2. Split the future into disjoint bins `(t_k, t_k+2], (t_k+2, t_k+4], …, (t_k+8, t_k+10]`.
3. For each bin, the evaluation set = triples **first** observed in that bin.
   `backtest` additionally restricts to pairs unconnected at the cutoff whose entities
   both already existed (otherwise no model could predict them) — the same novelty
   setting as hypothesis generation.
4. Repeat for several cutoffs (paper: 1990, 2000, 2010) and compare decay profiles.

## Metrics
| Metric | Meaning | Use |
|---|---|---|
| micro recall | share of future triples that were predicted in advance | headline |
| macro recall | per-relation recall averaged | detects whether rare relation types are covered |
| null recall | same budget, scores randomly drawn from the model's own score distribution per relation | the bar to beat |
| mean nDCG | per future pair: are the true relation types ranked first? | relation-type accuracy |
| covered | share of future triples whose pair was generated as candidate | upper bound on recall |

**Why no precision**: a prediction not observed in the window may simply be
undiscovered/unpublished. Precision is ill-defined in this positive–unlabeled setting;
the paper reports recall and ranking metrics instead. If a user insists, present
"precision lower bound" and explain why.

## Reading results
- Model ≫ null in every bin → the graph encodes predictive structure; proceed.
- Recall decays slowly across bins (paper ≈8% macro over 10 years) → long-horizon
  hypotheses retain signal. A steep collapse to the null → only short-term
  extrapolation works; focus on near-term, quickly testable hypotheses.
- Similar profiles across cutoffs → stable, era-independent behaviour (paper finding).
- High micro, low macro → dominated by frequent relations; check rare ones, consider
  the language-branch fusion which helps rare relations most.
- Low `covered` → candidate generation too narrow: `--hops 3`, raise `--hub-cap`,
  use seeds with `--seed-mode both`, or the graph is too sparse.
- `--top-frac` sets the prediction budget per relation; report it. Also show a
  budget sweep (e.g. 0.01, 0.05, 0.1, 0.2) when the user cares about operating points.

## How the back-test treats the fitted model
`H backtest --cutoff T --window W` fits the features model on `G(T−W) → (T−W, T]` and then
predicts from `G(T)` — the model never sees anything after `T`. Default `--model both` also
runs the fixed heuristic so you can see what fitting buys. Each back-test is saved as a
`runs/<id>_backtest_*` with `backtest.md/json`.

**Calibration in the large**: expected (sum of probabilities) vs. observed new triples within
`(T, T+W]`. Observed below expected is normal (unpublished ≠ false, and discovery rates drift);
a gap above ~3× means probabilities should be read as relative ranks, not absolute chances.

**Strongest standardized features** show what drives the model (e.g. high `pattern` and
`support` weights = relation-type rules and repeatedly reported neighbourhoods matter). Large
negative weights on correlated features (cn vs. Adamic–Adar) are collinearity, not a bug.

## Benchmark comparison (paper, 2020 cutoff, %)
| Model | Prec macro | Rec macro | F1 macro | Prec w | Rec w | F1 w | nDCG |
|---|---|---|---|---|---|---|---|
| Random | 6.1 | 6.9 | 3.8 | 18.2 | 6.1 | 8.4 | 35.2 |
| ComplEx | 5.3 | 8.3 | 5.8 | 17.2 | 19.7 | 18.2 | 47.2 |
| KNN | 33.0 | 26.9 | 28.5 | 60.9 | 58.5 | 59.1 | 74.6 |
| MLP | 39.5 | 26.8 | 28.8 | 67.0 | 60.2 | 61.5 | 86.8 |
| RuleBased (CN+PA) | 12.8 | 56.4 | 19.1 | 29.4 | 91.1 | 42.2 | 79.0 |
| tNodeEmbed | 40.6 | 27.4 | 29.2 | 68.9 | 62.4 | 63.5 | 87.6 |
| THiGER | 60.2 | 46.1 | 51.0 | 86.2 | 71.7 | 78.0 | 92.2 |
| THiGERLLM | 53.8 | 60.7 | 51.2 | 84.0 | 66.5 | 73.7 | 90.3 |

`hakken.py`'s features model is closest in spirit to the feature-based Science4Cast winners
and the RuleBased / THiGER-graph family; its numbers are not comparable to this table (different
data, candidate sets and budgets) — never claim THiGERLLM-level performance.

## Retrospective sanity check (cheap and convincing)
Pick 3–5 famous discoveries in the user's field with known publication years. Run
`H predict --cutoff <year-1>` and report the rank/percentile each received. This is an
intuitive complement to aggregate recall.
