# Why DA3 reaches zero Kenyon cells: the chain dies at hop 1, not at the mushroom body

**Date:** 2026-09-22 (06:25 Manila / 2026-09-21 22:25Z)
**Question:** item 3 of "Do next" in the 2026-09-22 project notes --
"Why DA3 reaches zero Kenyon cells and DA2 reaches five. Pre-existing, still unexplained,
possibly another connectome gap. Check their ALPN degrees against the raw Zenodo export, the
way the APL in-degree bug was caught."
**Tools:** `regime/alpn_kc_degree.py` (structural, ~2 min CPU, no sim),
`learn/gate_cs_channels.py --probe ORN_DA3,ORN_DA2` (the physiological confirmation).
**Raw:** `results/alpn_kc_degree.json`, `results/gate_cs_channels_g8.json` (both gitignored).

## Answer

Neither of the two explanations the handoff proposed is right. The synapses are **not** cut by
the export floor, and they are **not** absent from the raw connectome. DA3's ORNs contact all
four of its ALPNs, and carry **80 synapses** between them -- against DC2's 2356 and D's 3049.
The channel is wired and starved, and its projection neurons therefore never fire, so the
Kenyon-cell step is never reached at all.

| channel | ORNs | ALPNs | ORN->ALPN syn | x median | ALPN->KC KCs | measured KC frac | measured PN frac |
|---|---|---|---|---|---|---|---|
| **DA3** | 30 | 4 | **80** | 0.03x | 65 | **0.0000** | **0.0000** |
| **DA2** | 39 | 11 | **540** | 0.22x | 181 | 0.0008 (4 KC) | 0.0131 |
| DC2 (CS) | 20 | 4 | 2356 | 0.94x | 338 | 0.0603 (312 KC) | 0.0087 |
| D (CS) | 31 | 6 | 3049 | 1.22x | 324 | 0.0551 (285 KC) | 0.0160 |
| DA1 (probe) | 126 | 17 | 2940 | 1.18x | 600 | 0.0894 (463 KC) | 0.0437 |

Median over the 50 glomeruli that have ALPNs: 2502 hop-1 synapses, 349 KCs.

**The two channels fail for different reasons, and the PN column is what separates them.**

- **DA3 fails at hop 1.** PN active fraction is 0.0000. Its ALPNs never spike, so nothing
  downstream can be blamed. 80 synapses over 4 ALPNs is 20 each; a DC2 ALPN gets 589.
- **DA2 fails at hop 2.** PN active fraction is 0.0131, comparable to D's 0.0160 -- its
  projection neurons *do* fire. But its ALPN->KC mass is 1847 synapses against D's 5675, so
  the Kenyon cells stay subthreshold and only 4 ignite.

The handoff's "DA2 reaches five" is 4 here. Same picture, one cell of jitter.

## Why it is not the export floor, and not a gap

`brain_gpu.npz` is built at `min_syn = 5`. The script computes every number twice: once from
that matrix and once from `data/ext/proofread_connections_783.feather` at floor 1.

- **Export fidelity: 52/53 glomeruli** have `sim ALPN->KC count == raw count thresholded at
  min_syn` exactly. The one exception is VA1v (293 vs 295, two Kenyon cells). The export is
  faithful; `min_syn` is doing exactly and only what it says.
- **FLOORED: 0. ABSENT: 0.** No glomerulus loses its whole KC projection to the floor, and no
  glomerulus with ALPNs has zero ALPN->KC synapses in the raw data. Both of the handoff's
  candidate explanations are ruled out by measurement, not by argument.

Going from floor 5 to floor 1 adds KC targets everywhere (DA3 65 -> 110, DC2 338 -> 433) -- a
uniform ~25-30% of thin edges, not something specific to the weak channels.

## What this is and is not

- **It is** the same *shape* as the sugar -> PAM result
  (`reward-path/floor1_convergence_2026-09-21.md`): a pathway that is nominally present, whose
  every endpoint is connected, and which carries one to two orders of magnitude less synaptic
  mass than the channels that work. In both cases the honest statement is about v783 as
  reconstructed.
- **It is not** a claim that the animal's DA3 is silent. DA3 is a real, functional glomerulus.
  The positive control is already in hand and is in the same table: DC2, D and DA1 run through
  the identical ORN -> ALPN -> KC code path and ignite 312, 285 and 463 Kenyon cells. So the
  measurement works and the specific channel is thin.
- **It is not** a `min_syn` artifact, and this was the most likely boring explanation. It was
  checked first and it is dead.
- **Synapse mass is a prior, not a verdict; the gate run is the verdict.** Per-PN mass in
  particular is a bad discriminator: DA1 sits at 0.32x the median per PN and ignites more KCs
  than any other channel, because it has 17 ALPNs. Total hop-1 mass is the one that separates,
  and on the corrected run it flags exactly DA2 and DA3 — the two channels that do in fact fail
  at the gate — and nothing else.

## Consequences for the existing results

None of the rung A or rung 3 numbers move. DA3 was never a CS and never a probe -- the probe
set is DC2 / D / DA1, and all three sit at or above the median on both hops. If a future run
needs a fourth odour channel, pick it by the hop-1 column in
`results/alpn_kc_degree.json`, not by name.

**CORRECTED 2026-09-22, same day.** The first version of this section said `VM6l`, `VM6m` and
`VM6v` have ORNs (21, 33, 26) and no ALPN at all. That was a bug in this script, not anatomy.
The ORN naming is finer than the PN naming: all three receptor classes project to the single PN
type `VM6_adPN`, and `split("_")[0]` on the PN name yields `VM6`, which matched no `ORN_VM6`
channel. `VM6_adPN` receives 589 + 901 + 1339 = **2829 synapses** from those three ORN sets
(above the 2502 median) and reaches **335 Kenyon cells**. VM6 is a perfectly healthy channel.

The script now carries an explicit alias for it and, more importantly, prints every ALPN
glomerulus label with no matching `ORN_` channel, so the next naming mismatch surfaces instead
of being reported as missing anatomy. The remaining unmatched labels are `VP*` (thermo/hygro),
`CB*`, `M`, `MZ` and `Z` — multiglomerular and non-olfactory PNs with no single ORN channel,
which is expected.

With VM6 fixed, `NO_ALPN` is 0 and `STARVED_HOP1` is exactly DA2 and DA3.

## Reproduce

```
.venv/Scripts/python.exe regime/alpn_kc_degree.py
.venv/Scripts/python.exe learn/gate_cs_channels.py --probe ORN_DA3,ORN_DA2
```

The gate's PASS/FAIL is computed on the CS pair alone, so `--probe` channels are reported and
can never move the verdict. The gate still passes unchanged with the probes attached:
DC2 0.0603, D 0.0551, Jaccard 0.0566, 0 silent, 0 spontaneous.
