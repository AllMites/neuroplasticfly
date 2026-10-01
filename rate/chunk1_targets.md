# Chunk-1 target table: sugar -> PAM (path 2, PRD phase 2)

FIXED 2026-09-24, before any rate-model run touched a sugar, bitter, Fox, FDA or PAM neuron
(rate engine goldens use ORN_DM4 -> DM4_adPN only, flychess 43bd50f). Any change after the first
chunk-1 run is an amendment: dated, committed on its own, with its reason.

Targets are SIGNS, not magnitudes. Magnitudes in the sources are read off figure images, vary 2-3x
across Fox drivers (Christie preprint Fig 5D), and come from dF/F with no agreed rate mapping (PRD
open question). Thresholds ("up" = rate change above what) are set in the chunk-1 prereg, not here.

## Compartment readout sets (FlyWire v783 cell_type, all present; neuron counts in brackets)

Mapping from Aso 2014 via `learn/compartments.json`. A type that innervates two compartments counts
for both.

| compartment | types |
|---|---|
| g4 | PAM07 [18], PAM08 [45] |
| g5 | PAM01 [41], PAM15 [3] |
| b2 | PAM04 [32], PAM03 [10] |
| b'2 | PAM02 [16], PAM05 [21], PAM06 [30], PAM03 [10], PAM15 [3] |
| g3 | PAM12 [23] |
| b1 | PAM09 [9], PAM10 [15] |
| b'1 | PAM13 [12], PAM14 [16] |
| g1 | PPL101 [2] |
| g2 | PPL102 [2] |
| reward set R | g4, g5, b2, b'2 (union: PAM01-08, PAM15) |

Stimulus sets: sugar = `cell_sub_class` "sugar/water", bitter = "bitter" (`learn/condition.py:83-84`),
Fox = `CB0525` (`learn/christie_sweep.py:33`). **Assumption:** Christie 2026 gives no FlyWire type for
Fox; the chunk-1 prereg must cite where CB0525 comes from, or mark it an assumption in the paper.

## FIT rows (the only rows the optimiser sees)

Condition: Fox drive (the model's analogue of Christie's optogenetic Fox activation).

| id | compartment | target | source | panel | quote / note |
|---|---|---|---|---|---|
| F1 | g4 | up | Christie 2026 | pub Fig 6G; preprint Fig 5D | "Fox stimulation activated PAM-DANs innervating the MB γ4, γ5, β2, and β'2 compartments" |
| F2 | g5 | up | Christie 2026 | same | same |
| F3 | b2 | up | Christie 2026 | same | same |
| F4 | b'2 | up | Christie 2026 | same | same |
| F5 | b1 | none | Christie 2026 (preprint 5D, all 3 drivers flat) + Siju 2020 Fig 4B (coef n.s.) | as listed | the only null that holds in two papers and no paper contradicts; stops the degenerate "every PAM up" fit |

## HELD-OUT tests (never in the loss; pass/fail thresholds set in the chunk-1 prereg)

| id | condition | test | source | why held out |
|---|---|---|---|---|
| H1 | sugar GRN drive | g4, g5 up | Cohn 2015 Fig 2B (fasted 20-26 h, ingestion); Christie Fox result | stimulus transfer: the GRN -> Fox part of the path is not in the fit condition |
| H2 | sugar GRN drive | b2, b'2 up | Siju 2020 Fig 4B (sucrose-vs-quinine coef > 0, starved 24 h) | as H1; second, independent paper |
| H3 | bitter GRN drive | reward set R NOT up | Siju 2020 Fig 4B (R coefs sucrose-biased) | "bitter excluded"; LIF currently recruits 5-12 PAMs for bitter |
| H4 | sugar GRN drive, Fox silenced | response of R drops vs H1/H2 | Christie 2026 (Fox silencing abolishes Gr64f learning; BEHAVIOUR only, no imaging) | causal test; weaker evidence type, stated as such |
| H5 | Fox drive | g3 none | Christie preprint 5D (flat, all 3 drivers) | held-out null, same condition as the fit |
| H6 | bitter vs sugar drive | g1, g2 (PPL1) higher for bitter than sugar | Siju 2020 Fig 4B (coef < 0) | valence contrast outside the PAMs |

## DESCRIPTIVE only (reported, never pass/fail)

| id | observation | source | why not a test |
|---|---|---|---|
| X1 | g2, g3 down with sugar | Cohn 2015 Fig 2B | g3 conflicts (Christie null, Siju n.s.); Cohn's ingestion also changes motor state, and quiescence alone gives the same pattern |
| X2 | b'1 under Fox drive | Christie preprint 5D | one of 3 drivers responds; not in the paper's text |
| X3 | a1, b1 carry nutrient/LTM reward (activation sufficient) | Huetteroth 2015 Fig 3K; Yamagata 2015 Fig 4C/F | behaviour, post-ingestive; chunk 1 is taste only |
| X4 | b'2am + g4 required for sweet (arabinose) memory | Huetteroth 2015 Fig 2E/H/I | behaviour; supports R, not a response target |
| X5 | hunger raises a1, b1, b'2, g4 odour responses | Siju 2020 Fig 5A | model has no hunger state |

## Not used, and why

- Magnitudes (all sources): read off images; driver-dependent spread; no dF/F -> rate mapping.
- Christie's connectome simulation (Shiu LIF, Wsyn 0.37-0.39): not a recording. Their Table S1B PAM
  count jumps 0/8/6/8/0/0/200/... across drive rates, which looks like runaway all-or-none activity.
- Liu 2012 Nature: full text not accessed. Lin 2014: imaging is water only.
- Alpha / alpha' lobes under Fox: not imaged by Christie.

## Caveats the paper must carry

1. Christie imaging is an ex vivo explant (saline with 8 mM trehalose + 10 mM glucose), hunger state
   not stated. Fitting to it fixes an unstated state.
2. Siju "up" is a sucrose-vs-quinine regression sign, not a raw sucrose response; ~30 % of taste
   datasets were discarded as inactive.
3. Five fit rows is very few. The type-level parameterisation, the held-out battery, and the >= 5-seed
   ensemble exist because of this (PRD risk "too few fit targets").

## Data that could replace signs with numbers

- Siju 2020 taste light-field data + analysis: https://github.com/sophie63/Siju2020 (public; not
  downloaded yet). Would give per-compartment sucrose/quinine traces for H2/H3/H6. Using it for any
  FIT row is an amendment.
- Christie 2026 raw imaging: on request (Shao lab). Cohn 2015: unknown (supplement not read).
- Requests are the user's call.

## Sources

- Christie KW et al. 2026 Curr Biol 36(4). DOI 10.1016/j.cub.2025.12.058. PMC12869359; preprint bioRxiv 10.1101/2025.10.31.685871.
- Cohn R, Morantte I, Ruta V 2015 Cell 163:1742.
- Siju KP et al. 2020 Curr Biol 30(11):2104.
- Huetteroth W et al. 2015 Curr Biol 25:751.
- Yamagata N et al. 2015 PNAS 112:578.
- Aso Y et al. 2014 eLife 3:e04577 (compartment map).

Extraction: subagent read of full texts 2026-09-24 (Christie published text via NCBI BioC + preprint
figures; Cohn author manuscript via BioC; Siju and Yamagata via publisher pages). Christie published
Fig 6G / S5H images and Cohn Table S1 were not seen.
