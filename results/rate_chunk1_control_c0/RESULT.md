# Chunk-1 unfitted control on the chunk-0 base — RESULT

Procedure `194be67` + prereg note `2c0b86d`. One arm C0 (rate/regress/c0_base.json). No fitting.

**LABEL: FAILS-UNFITTED** (stable: <= 0.1 % of neurons at r_max; max 150 Hz = the driven GRNs/Fox).
Rows identical to arm A: F5, H3, H5 pass (nothing responds); F1-F4, H1, H2, H4, H6 fail.
Reward-set PAMs (R, 216) stay at 0.000 Hz in every condition.

## Where sugar stops (descriptive, no label)

| Condition | Fox | FDA-I | FDA-II | reward PAMs |
|---|---|---|---|---|
| FOX 150 Hz | 150 (driven) | 6.6 | 7.0 | 0 |
| SUGAR 150 Hz | 10.6 | 1.0 | 0.6 | 0 |
| SUGAR, Fox silenced | 0 | 0.9 | 0.2 | 0 |
| BITTER 150 Hz | 0 | 0 | 0 | 0 |

- Anatomy (floor-5 W): **sugar GRNs make 0 direct synapses onto Fox.** Fox's 1,320 input synapses
  come from 88 partners; sugar reaches Fox only via two-hop interneurons (e.g. CB0366, AN_GNG_30, CB0501).
- So on this base the path has two weak stages: sugar -> (interneurons) -> Fox (10.6 of a possible 150 Hz),
  and Fox/FDA -> PAM (0 Hz even with Fox driven at 150 Hz). The FDA -> PAM stage is the one chunk 1 owns.
- Silencing Fox barely changes FDA-I under sugar (1.0 -> 0.9 Hz): sugar reaches FDA-I mostly not via Fox.
- Stress check in rate/test_suite.py step 7 (reward pools: no threshold, gain 1e4) again recruits F1/F2/F4
  and sugar H1 but breaks H3 (bitter recruits reward PAMs), as on arm A.

rate/regress/c1_record.json re-recorded from these rows; suite GREEN on the chunk-0 base.
