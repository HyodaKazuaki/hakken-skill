# Domain adaptation

Hakken is domain-general: anything expressible as typed, directed relations between
concepts with publication dates works. Below are starting schemas; refine with the
user. Keep 5–30 relations, mechanistic where possible, with parents for polarity pairs.

## Contents
- [Biomedicine (paper's main domain)](#biomedicine)
- [Drug discovery / repurposing](#drug-discovery)
- [Materials science](#materials-science)
- [Chemistry](#chemistry)
- [Agriculture](#agriculture)
- [Ecology & environment](#ecology)
- [Social & behavioural science](#social-science)
- [Computer science / AI](#cs-ai)
- [Physics & engineering](#physics-engineering)
- [Choosing seeds and cutoffs](#choosing-seeds-and-cutoffs)

## Biomedicine
Entities: gene/protein, chemical/drug, disease, phenotype, pathway, cell type, organism.
Relations used in the paper (after expert merging, Table E3): `INDUCES`,
`IS_A_BIOMARKER_FOR`, `INHIBITS`, `TREATS`, `IS_DIFFERENTIATED_BY`, `IS_A_TARGET_FOR`,
`IS_A`, `METABOLIZES`, `CONTAINS`, `PREVENTS`, `INCREASES_DEGRADATION_OF`,
`AFFECTS_DEGRADATION_OF`, `{INCREASES,DECREASES,AFFECTS}_TRANSCRIPTIONAL_ACTIVITY_OF`,
`{INCREASES,DECREASES,AFFECTS}_EXPRESSION_OF`, `INHIBITS_GROWTH_OF`,
`INHIBITS_PROLIFERATION_OF`, `{INCREASES,DECREASES,AFFECTS}_ABUNDANCE_OF`.
Sources: PubTator3, SemMedDB, PubMed. Normalise with HGNC, MeSH, MONDO, ChEBI, UniProt.
Validation designs: knockdown/overexpression + qPCR/western; reporter assays; inhibitor
treatment + expression readout (as in the TP53→BAMBI, RAF1→TNF confirmations).

## Drug discovery
Same schema, seeds = targets or indications. Use the calibrated score as a
**knowledge score** alongside ADMET when ranking candidates (paper §5). Typical
queries: `(drug, TREATS, ?)`, `(?, IS_A_TARGET_FOR, disease)`, `(drug, INHIBITS, gene)`.

## Materials science
Entities: material/composition, structure/phase, property, synthesis method, dopant,
application, characterisation technique.
Relations: `HAS_PROPERTY`, `ENHANCES_PROPERTY`, `DEGRADES_PROPERTY`, `SYNTHESIZED_BY`,
`DOPED_WITH`, `STABILIZES`, `USED_IN`, `CATALYZES`, `FORMS_PHASE`, `SUBSTITUTES_FOR`.
Polarity parent: `AFFECTS_PROPERTY` over enhance/degrade.
Sources: OpenAlex/Crossref abstracts, Materials Project (normalisation), patents.
Validation: computational screening (DFT), then synthesis + characterisation.

## Chemistry
Entities: compound/class, reaction type, catalyst, solvent, condition, product property.
Relations: `REACTS_WITH`, `CATALYZES`, `INHIBITS`, `YIELDS`, `IS_PRECURSOR_OF`,
`INCREASES_YIELD_OF`, `DECREASES_YIELD_OF` (parent `AFFECTS_YIELD_OF`), `SOLVATES`.
Sources: PubChem, ChEBI, reaction literature. Validation: small-scale reaction screens.

## Agriculture
Entities: crop/cultivar, practice (rotation, intercropping, mulching), soil property,
pest/pathogen, beneficial organism, input (fertiliser, compost), outcome (yield, quality).
Relations: `INCREASES_YIELD_OF`, `DECREASES_YIELD_OF`, `SUPPRESSES`, `ATTRACTS`,
`IMPROVES`, `DEPLETES`, `COMPANION_OF`, `SUSCEPTIBLE_TO`, `RESISTANT_TO`.
Sources: AGROVOC (normalisation), CAB, historical texts (the paper used 19th-century
market-gardening texts — old corpora work if dated by publication).
Validation: plot trials, greenhouse experiments.

## Ecology
Entities: species, habitat, environmental factor, trait, ecosystem function.
Relations: `PREYS_ON`, `POLLINATES`, `PARASITIZES`, `COMPETES_WITH`, `FACILITATES`,
`INCREASES_ABUNDANCE_OF`, `DECREASES_ABUNDANCE_OF`, `INHABITS`, `TOLERATES`.
Sources: GloBI, GBIF, OpenAlex. Validation: field surveys, mesocosms.

## Social science
Entities: construct/variable, intervention, population, outcome, policy.
Relations: `INCREASES`, `DECREASES` (parent `AFFECTS`), `MODERATES`, `MEDIATES`,
`PREDICTS`, `CORRELATES_WITH`. Caution: effects are context-dependent — record the
population as part of entity names or as a condition note.
Validation: pre-registered studies, secondary data analysis.

## CS / AI
Entities: method/architecture, task, dataset, metric, technique/component, hardware.
Relations: `IMPROVES_ON` (task), `APPLIED_TO`, `EVALUATED_ON`, `COMBINED_WITH`,
`EXTENDS`, `REDUCES_COST_OF`, `OUTPERFORMS`. Time unit can be months/quarters
(use `yyyymm` as integer "year" and set `--tau` / `--bin` accordingly).
Sources: arXiv API, OpenAlex, Semantic Scholar. Validation: run the experiment.

## Physics / engineering
Entities: phenomenon, material/system, parameter, method, device.
Relations: `EXHIBITS`, `INDUCES`, `SUPPRESSES`, `MEASURED_BY`, `ENABLES`, `TUNES`.

## Choosing seeds and cutoffs
- Seeds: 20–2,000 entities defining the user's area (expert list, top entities for a
  topic query, or a pathway/material family). `--seed-mode both` scores all seed pairs
  (quadratic); `any` scores 2-hop neighbourhoods touching a seed.
- Back-test cutoffs: at least 10 units before the latest data so 5 bins of 2 can be
  evaluated; use 2–3 cutoffs to check the decay profile is stable.
- Prediction cutoff: the latest complete year (drop the current partial year).
