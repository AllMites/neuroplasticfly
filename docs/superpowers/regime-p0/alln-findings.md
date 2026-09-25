<!-- AUTHORITATIVE record of the ALLN probes that followed handoff.md. `results/` is
gitignored; every number here was measured on the host on 2026-09-19 with the scripts
named, and can be regenerated with them (each is one command, GPU, under 10 min). -->

# Regime stability -- ALLN findings

Date: 2026-09-19. Continues `handoff.md` (next steps 1, 2, 4). Base HEAD `90f4f5c`.

Scripts, all read-only against the shipped brain (in-memory weight variants on one
`GpuSim`, nothing on disk changes, no `gpu_sim.py` constant touched):

| script | question |
|---|---|
| `regime/layers.py` | per-layer Jaccard, same seed both channels |
| `regime/alln_probe.py` | zero / flip the 134 excitatory ALLNs |
| `regime/alln_sweep.py` | exc-ALLN gain x ORN drive |
| `regime/alln_probe2.py` | which exc-ALLN edges; global gain with the loop gone |
| `regime/alln_multiglom.py` | two 8-glomerulus odours, exc-ALLN as-is vs flipped |

---

## 1. Seed confound fixed; headline unchanged

`regime/probe.py:measure()` now runs every channel on `seed0` (the counter RNG hashes
`(seed, step, neuron)`, so one seed = one shared noise stream). 60 Hz, seeds 0/2/4:

| layer | handoff (confounded) | now |
|---|---|---|
| ALLN | 0.976 | 0.983 |
| ALPN | 0.940 | 0.934 |
| KC | 0.964 | 0.965 |
| central | 0.946 | 0.946 |

Per-seed spread is 0.003. Every number in `handoff.md` stands.

## 2. The merge is the excitatory-ALLN -> ALPN broadcast

Synapse sign is per neuron from FlyWire `top_nt` (GABA and glutamate inhibitory,
everything else excitatory; no neuron has mixed-sign out-edges). **134 of 429 ALLNs
are excitatory** (top types `lLN1_bc` 30, `lLN2X12` 13, `lLN2X03` 6, `v2LN4` 6). Their
15,641 out-edges split 6,294 -> ALLN, 7,172 -> ALPN, 2,175 -> other.

Input onto the 345 ALPNs that fire for DA1 without a DA1 ORN synapse, from active
presynaptic neurons, summed `W_data`: ALLN +89,611 / -65,740; ALPN +23,767; everything
else under 5,000 excitatory. Only 13 ALLNs receive a DA1 ORN synapse; 231 fire.

60 Hz, seed 0, `act/act Jaccard`:

| variant | ALLN | ALPN | KC | central |
|---|---|---|---|---|
| base | 231/229 0.98 | 360/345 0.94 | 1698/1705 0.96 | 5114/5091 0.94 |
| exc-ALLN out-edges zeroed | 4/51 0.02 | 7/10 0.00 | 0/13 0.00 | 13/661 0.00 |
| exc-ALLN out-edges negated | 4/33 0.00 | 7/10 0.00 | 0/9 0.00 | 13/671 0.00 |
| all ALLN out-edges zeroed | 8/89 0.04 | 11/13 0.00 | 0/27 0.00 | 28/577 0.01 |
| zero only exc ALLN -> ALLN | 173/169 0.97 | 352/344 0.94 | 1345/1386 0.94 | 4432/4500 0.94 |
| zero only exc ALLN -> ALPN | 5/181 0.03 | 7/8 0.00 | 0/0 0.00 | 14/737 0.01 |
| zero only exc ALLN -> other | 230/229 0.98 | 352/339 0.94 | 1701/1696 0.96 | 5101/5085 0.94 |

The LN -> LN recurrence is not the merge (removing it leaves 0.94). The LN -> PN
broadcast is: remove those 7,172 edges and the two odours never touch. With the loop
off, DA1 drives exactly 7 ALPNs: the DA1 uniglomerular PNs (15 `DA1_lPN` + 2 `DA1_vPN`
receive >= 5 DA1 ORN synapses; 7 cross threshold). The AL output is then correct.

## 3. No gain window on the loop; no rescue by global gain

Exc-ALLN out-edges scaled, ORN drive swept (seed 0):

| scale | 60 Hz | 120 Hz | 300 Hz |
|---|---|---|---|
| 0.00 | ALPN 7/10 0.00, KC 0/13 | ALPN 11/12 0.00, KC 0/76 | ALPN 16/20 0.00, KC 2/169 |
| 0.25 | ALPN 8/72 0.00, KC 0/75 | ALPN 13/68 0.00, KC 0/112 | ALPN 19/55 0.01, KC 3/172 |
| 0.50 | ALPN 8/206 0.00 | ALPN 214/203 0.85, KC 0.59 | ALPN 217/196 0.79, KC 0.42 |
| 0.75 | ALPN 307/292 0.93, KC 0.92 | 0.92 / 0.88 | 0.88 / 0.81 |

Bistable at every gain; the attractor shrinks with gain but stays shared. Loop off plus
global weight x1.5 / x2 / x3: ALPN stays at 8-28 for DA1 while central climbs to
2,587-4,297 with Jaccard 0.53-0.59 -- central ignites through some other loop, bypassing
the AL. Global gain was ruled out in the handoff with the AL loop present; it is still
ruled out with it absent.

## 4. A single glomerulus cannot reach a Kenyon cell, and that is the connectome

DA1's 17 PNs synapse onto 600 KCs, but only 5 KCs receive >= 3 of those PNs and 60 get
>= 25 synapses (7 mV at `W_SYN`). DM4: 597 KCs touched, 0 with >= 3 PNs. Median PN -> KC
edge is 13-14 synapses (3.6-3.9 mV, half threshold). A KC needs several PNs from
different glomeruli coincident. **Single-glomerulus Jaccard at KC was never a test of
anything downstream of the AL.** Real odours drive 10-30 glomeruli.

## 5. Multi-glomerular odours, exc-ALLNs flipped inhibitory: separated and alive

Odour A = DA1 VA1d VL1 DL3 DM1 VA2 DM3 DM2 (313 ORNs), odour B = DM4 VM5d VM4 DL1 V VA6
DM3 DM2 (253 ORNs), two glomeruli shared, ORN Jaccard 0.12. `act/act Jaccard`:

| variant | Hz | ALLN | ALPN | KC | central |
|---|---|---|---|---|---|
| base | 60 | 229/233 0.97 | 350/349 0.94 | 1796/1812 0.92 | 5215/5213 0.93 |
| flipped | 20 | 31/27 0.38 | 16/35 0.21 | 4/0 0.00 | 613/156 0.09 |
| flipped | 60 | 36/48 0.45 | 38/50 0.24 | 77/32 0.02 | 784/796 0.48 |
| flipped | 120 | 46/53 0.43 | 47/57 0.21 | 190/117 0.05 | 975/943 0.36 |
| flipped, seed 2 | 60 | 36/45 0.45 | 37/46 0.19 | 87/8 0.00 | 840/345 0.12 |
| flipped, seed 4 | 60 | 37/48 0.47 | 34/49 0.20 | 86/24 0.01 | 842/782 0.49 |

KC 1.5% / 0.2-0.6% active with Jaccard <= 0.02, central alive at 2.4-2.6%, no
ignition. ALPN Jaccard 0.19-0.24 against a 0.12 ORN overlap. Odour B's KC count
(8-32) sits near threshold and swings with seed; odour A's (77-87) does not.

## What this does and does not say

- The sim's whole-brain ignition at the AL is caused by 134 top_nt-cholinergic ALLNs
  broadcasting excitation onto PNs of glomeruli they were not driven from. Flip their
  sign and the AL separates odours through KC. This is a targeted change to 134 neurons'
  sign, not a new mechanism, and it is testable against the literature.
- Literature (Chou et al. 2010; Das et al. 2011; Seki et al. 2010): the lateral LN
  lineage is mostly GABAergic, with a smaller cholinergic (eLN) set and a glutamatergic
  lineage. 31% cholinergic is high but not provably wrong. Whether `lLN1_bc` and the
  `lLN2X*` types are really cholinergic in FlyWire's annotation, and with what
  confidence, was NOT checked (no nt-confidence field is exported).
- What is NOT fixed: with the loop off the feedforward path is weak (single glomerulus
  -> 0 KC; 8 glomeruli -> 0.2-1.5% KC), and central still carries a second, AL-independent
  ignition loop that appears under global gain >= x2.
- Not measured: ignition time course (needs per-step recording); APL / SFA arms on top
  of the flipped brain; plasticity (criterion 2).

## 6. The sign question, answered: it is the real eLNs

`regime/nt_conf.py` builds `data/nt_conf.npz` from the public FlyWire v783 annotation TSV
(sorted by root_id = neuron order, verified on four columns row for row). For the 134
excitatory ALLNs:

| group | n | top_nt | conf median | known_nt |
|---|---|---|---|---|
| all 134 | 134 | ACh 80, 5-HT 40, DA 12, OA 2 | 0.36 (brain-wide ACh 0.72); 0 at >= 0.8 | ACh 44 |
| known ACh (Shang et al. 2007, immuno) | 44 | mostly 5-HT / DA predicted | 0.31 | `lLN1_bc` 30, `lLN2X03` 6, `lLN2T_b/c` 8 |
| ACh-predicted, no known_nt | 69 | ACh | 0.40 | `lLN2X12`, `v2LN4`, ... |
| aminergic-predicted, no known_nt | 21 | 5-HT / DA / OA | 0.34 | - |

The predictions are noise (a local neuron is not serotonergic at 0.34), but the sign in
`W_data` is right for at least the 44: Shang et al. 2007 immuno-confirmed cholinergic
eLNs. `regime/alln_known.py`, 8-glomerulus pair, 60 Hz, seeds 0/2, `ALPN Jaccard / KC`:

| flip to inhibitory | ALPN | KC |
|---|---|---|
| all 134 | 0.24 / 0.02 | separated |
| the 44 known eLNs only | 0.20 / 0.01-0.03 | separated |
| the 21 noise-predicted without the 44 | 0.83 / 0.51 | ignites |
| the 90 without known_nt | 0.78 / 0.42 | ignites |

**The 44 immuno-confirmed eLNs carry the merge.** Flip them and the AL separates; flip
everything except them and it still ignites. Not a data correction.

Gain on their edges instead of a flip (seeds 0/2 agree to 0.01):

| 44 eLN edges | x0 | x0.25 | x0.5 |
|---|---|---|---|
| -> ALPN only (3,802 edges) | ALPN 0.85 | 0.86 | 0.91 |
| -> all (7,318) | - | 0.87 | - |
| -> non-ALPN only zeroed | 0.93 | - | - |

Zeroing is not enough; negating is. What separates is inhibition arriving where their
excitation did, i.e. the other 90 excitatory ALLNs still ignite the PNs once the 44 stop
holding them back. No partial-gain window exists on this axis either.

Physiology: Yaksi & Wilson 2010 and Huang et al. 2010 show eLN -> PN lateral excitation is
ELECTRICAL (gap junctions; abolished by a shakB mutation, untouched by blocking chemical
transmission). This LIF delivers every eLN synapse as a spike-triggered `W_SYN` x count
chemical EPSP, which is the wrong primitive for those 3,802 edges. That is the spec item.

## 7. ELN_NEGATE bridge switch (committed)

`gpu_sim.ELN_NEGATE` (default `False`) is an authored bridge: when on, `GpuSim.__init__`
negates the outgoing weights (7,318 rows) of the 44 immuno-confirmed cholinergic
antennal-lobe eLNs on the device copy of the weight tensor, leaving the OFF path
bit-identical. `regime/alln_multiglom.py --negate --seeds 0 2 4`, two 8-glomerulus odours
(313 / 253 ORNs, 2 shared glomeruli), `act A / act B  J`:

| variant | Hz | seed | ALLN | ALPN | KC | central |
|---|---|---|---|---|---|---|
| base | 20 | 0 | 228/232 0.97 | 350/345 0.96 | 1734/1731 0.96 | 5135/5123 0.96 |
| base | 20 | 2 | 231/232 0.99 | 347/345 0.96 | 1730/1732 0.96 | 5125/5124 0.95 |
| base | 20 | 4 | 228/234 0.97 | 348/348 0.96 | 1735/1728 0.96 | 5125/5108 0.96 |
| base | 60 | 0 | 229/233 0.97 | 350/349 0.94 | 1796/1812 0.92 | 5215/5213 0.93 |
| base | 60 | 2 | 231/232 0.97 | 349/348 0.95 | 1797/1812 0.92 | 5225/5215 0.93 |
| base | 60 | 4 | 230/235 0.97 | 348/348 0.94 | 1803/1818 0.91 | 5252/5264 0.93 |
| base | 120 | 0 | 233/234 0.95 | 358/348 0.90 | 1886/1891 0.87 | 5392/5342 0.90 |
| base | 120 | 2 | 234/233 0.95 | 357/349 0.91 | 1883/1894 0.87 | 5380/5317 0.90 |
| base | 120 | 4 | 234/235 0.95 | 362/351 0.90 | 1886/1890 0.87 | 5404/5362 0.90 |
| ELN_NEGATE | 20 | 0 | 44/49 0.52 | 23/37 0.20 | 4/0 0.00 | 618/187 0.11 |
| ELN_NEGATE | 20 | 2 | 46/48 0.47 | 25/42 0.20 | 12/0 0.00 | 684/384 0.31 |
| ELN_NEGATE | 20 | 4 | 45/63 0.57 | 28/42 0.27 | 10/0 0.00 | 661/732 0.65 |
| ELN_NEGATE | 60 | 0 | 70/70 0.52 | 62/58 0.20 | 119/20 0.01 | 948/433 0.13 |
| ELN_NEGATE | 60 | 2 | 68/76 0.60 | 61/55 0.20 | 112/28 0.03 | 870/444 0.15 |
| ELN_NEGATE | 60 | 4 | 65/80 0.58 | 55/59 0.21 | 137/57 0.06 | 964/893 0.47 |
| ELN_NEGATE | 120 | 0 | 75/107 0.52 | 71/76 0.18 | 208/165 0.10 | 1121/1206 0.38 |
| ELN_NEGATE | 120 | 2 | 71/101 0.59 | 73/75 0.20 | 205/188 0.11 | 1101/1215 0.39 |
| ELN_NEGATE | 120 | 4 | 75/93 0.58 | 69/73 0.17 | 205/165 0.09 | 1106/1149 0.39 |

Verdict against the plan targets, all at 60 Hz unless stated (KC n = 5,177, central
n = 32,383):

- **ALPN Jaccard <= 0.35: PASS.** 0.20 / 0.20 / 0.21 across seeds 0/2/4 (base: 0.94-0.95).
- **KC Jaccard <= 0.20: PASS.** 0.01 / 0.03 / 0.06 (base: 0.91-0.92).
- **KC active 2-10% of 5,177 for BOTH odours: MISS.** Odour A is in band at
  119 = 2.30%, 112 = 2.16%, 137 = 2.65%. Odour B is below the 2% floor at every seed:
  20 = 0.39%, 28 = 0.54%, 57 = 1.10%. At 20 Hz odour B has zero active KCs. The negated
  brain separates the odours by making the second one nearly silent at KC, which is not
  the sparse-but-present code the target asks for.
- **central 1-5% of 32,383 active: PASS.** Odour A 948 = 2.93%, 870 = 2.69%,
  964 = 2.98%; odour B 433 = 1.34%, 444 = 1.37%, 893 = 2.76%.
- **120 Hz central < 8%: PASS.** Odour A 1121 = 3.46%, 1101 = 3.40%, 1106 = 3.41%;
  odour B 1206 = 3.72%, 1215 = 3.75%, 1149 = 3.55%.

Four of five targets pass; KC occupancy misses on odour B. Bridge, superseded by
GAP_COUPLE (plan phase 2).

## 8. GAP_COUPLE primitive landed (sweep is phase 3)

`GAP_COUPLE` replaces the 44 known eLNs' 3,802 chemical edges onto ALPNs with a
per-step resistive coupling delivered straight into `v`: `dv_j += k (v_i - v_j)`
and the mirror onto `i`, `k = GAP_COUPLE * synapse_count`, applied only to
non-refractory ends. The chemical entries for those edges are zeroed on the
device copy and excluded from the homeostatic normalisation totals; the host
matrix is untouched and the default (0.0) is bit-identical to everything before.

**Stability bound.** `kmax = 0.000510725`, from a maximum per-neuron summed
count of 1958. This is a NUMERICAL bound (it keeps the explicit step a convex
combination), not a physiological one.

**Heterogeneity.** Per-target summed counts over the 337 gap-target ALPNs:
median 80, p90 399, max 1958 — 24x the median. One scalar `GAP_COUPLE`
over-couples the heaviest PNs long before it moves the median, which is why
per-target normalisation is an open phase-3 question.

**Negative result at kmax*0.5.** The plan's smoke point was wrong. At a summed
coefficient of 0.5 per step a PN relaxes to the eLN pool voltage with a ~2-step
(0.2 ms) time constant, ~85x faster than TAU_M, i.e. a hard clamp rather than a
coupling. It kills the brain: ALPN 11/11 active, KC 0.0000.

**Strength sweep** (two 8-glomerulus odours, 60 Hz, seed 0; ALPN Jaccard / KC
active fraction). Baseline with the chemical path intact is 0.942 / 0.347.

| GAP_COUPLE | ALPN Jaccard | KC active | ALPN active |
|---|---|---|---|
| kmax * 1e-6 (chemical zeroed, coupling ~nil) | 0.840 | 0.053 | 183/196 |
| kmax * 0.01 | 0.784 | 0.038 | 165/181 |
| kmax * 0.1 | 0.586 | 0.017 | 122/146 |
| kmax * 0.5 | 0.100 | 0.000 | 11/11 |

**Reading.** The two effects switch on together, so the `1e-6` row isolates
them: removing the chemical eLN broadcast ALONE does most of the work
(KC 0.347 -> 0.053, ALPN Jaccard 0.942 -> 0.840). The coupling itself then adds
separation more slowly and costs KC occupancy the whole way. Whatever the
working point turns out to be, most of the regime change is attributable to
deleting the chemical broadcast, not to the electrical primitive replacing it —
which is a claim constraint for any reel or writeup.

**Normalisation decision.** `_input_scale` gained an `exclude` argument. Without
it, the zeroed gap edges still counted toward their targets' incoming budget, so
the 337 target ALPNs lost a median 11.6% and up to 36.3% of the chemical input
that actually arrives. Measured both ways; the exclusion is asserted in
`regime/test_gap.py` to 1e-2 relative.

**Throughput** (RTX 5070 Ti, `GAP_COUPLE = kmax*0.5` for the ON column).

| | OFF | ON, dense `[N,B]` dv | ON, edge-level |
|---|---|---|---|
| B=8 | 4.6 sims/s | 3.0 | 3.0 |
| B=64 | 21.2-21.5 sims/s | 15.1 | 17.4 |

The first ON form built, zeroed and masked a dense `[N, B]` buffer per step to
move 381 rows. The edge-level form scatters `[2E, B]` straight into `v` with the
refractory mask gathered per edge end. That recovers most of the gap at B=64
(-29% -> -19% vs OFF) and none of it at B=8, where per-step launch overhead
dominates. The plan's "within 10% of baseline" target was not met and has been
replaced by "throughput recorded".

Working point is phase 3.

## 9. GAP_COUPLE sweep, criterion 1 (phase 3)

`regime/gap_sweep.py`, nine arms x 2 rates x 3 seeds on the standard two-odour
multi-glomerular probe (313 / 253 ORNs, 2 shared glomeruli), every arm a fresh
`GpuSim`, nothing but `GAP_COUPLE` / `GAP_NORM` / `ELN_NEGATE` moved. `kmax` is
derived in the script (0.000510725, never hardcoded). The recorded run below was made
at commit `f07cc8e` (before the harness gained its provenance header and JSON sibling in
`55b7481`); `results/gap_sweep.txt` is gitignored, so this section is the record. New this phase:
`gpu_sim.GAP_NORM` (AUTHORED, default `False`) divides each gap coefficient by its
TARGET's summed synapse count, so every coupled PN carries a summed per-step
coefficient of exactly `GAP_COUPLE` - which then reads as S, with the
physiological band S ~ 0.0005-0.001. Output copied verbatim from
`results/gap_sweep.txt` (gitignored):

```text
kmax 0.000510725   KC n 5177   central n 32383   rates [60, 120]   seeds [0, 2, 4]
arm              hz              ORN             ALLN             ALPN               KC          central  seed
base             60    313/ 253 0.12    229/ 233 0.97    350/ 349 0.94   1796/1812 0.92   5215/5213 0.93  seed0
base             60    313/ 253 0.12    231/ 232 0.97    349/ 348 0.95   1797/1812 0.92   5225/5215 0.93  seed2
base             60    313/ 253 0.12    230/ 235 0.97    348/ 348 0.94   1803/1818 0.91   5252/5264 0.93  seed4
base            120    313/ 253 0.12    233/ 234 0.95    358/ 348 0.90   1886/1891 0.87   5392/5342 0.90  seed0
base            120    313/ 253 0.12    234/ 233 0.95    357/ 349 0.91   1883/1894 0.87   5380/5317 0.90  seed2
base            120    313/ 253 0.12    234/ 235 0.95    362/ 351 0.90   1886/1890 0.87   5404/5362 0.90  seed4
delete           60    313/ 253 0.12    199/ 204 0.94    183/ 196 0.84    275/ 217 0.33   1877/1939 0.70  seed0
delete           60    313/ 253 0.12    200/ 202 0.93    182/ 193 0.87    253/ 213 0.32   1881/1911 0.72  seed2
delete           60    313/ 253 0.12    199/ 206 0.93    181/ 194 0.82    269/ 206 0.30   1896/1880 0.71  seed4
delete          120    313/ 253 0.12    201/ 206 0.93    191/ 193 0.74    343/ 293 0.21   2057/2018 0.60  seed0
delete          120    313/ 253 0.12    201/ 205 0.92    187/ 194 0.74    345/ 285 0.21   2056/2016 0.59  seed2
delete          120    313/ 253 0.12    203/ 206 0.91    192/ 196 0.75    338/ 309 0.21   2099/2079 0.61  seed4
negate           60    313/ 253 0.12     70/  70 0.52     62/  58 0.20    119/  20 0.01    948/ 433 0.13  seed0
negate           60    313/ 253 0.12     68/  76 0.60     61/  55 0.20    112/  28 0.03    870/ 444 0.15  seed2
negate           60    313/ 253 0.12     65/  80 0.58     55/  59 0.21    137/  57 0.06    964/ 893 0.47  seed4
negate          120    313/ 253 0.12     75/ 107 0.52     71/  76 0.18    208/ 165 0.10   1121/1206 0.38  seed0
negate          120    313/ 253 0.12     71/ 101 0.59     73/  75 0.20    205/ 188 0.11   1101/1215 0.39  seed2
negate          120    313/ 253 0.12     75/  93 0.58     69/  73 0.17    205/ 165 0.09   1106/1149 0.39  seed4
scalar x0.003    60    313/ 253 0.12    198/ 203 0.93    179/ 192 0.83    246/ 184 0.31   1821/1832 0.71  seed0
scalar x0.003    60    313/ 253 0.12    199/ 201 0.92    180/ 188 0.84    232/ 195 0.33   1790/1825 0.71  seed2
scalar x0.003    60    313/ 253 0.12    198/ 204 0.92    177/ 190 0.81    256/ 199 0.32   1820/1769 0.69  seed4
scalar x0.003   120    313/ 253 0.12    199/ 205 0.92    179/ 191 0.71    316/ 284 0.20   1948/1960 0.59  seed0
scalar x0.003   120    313/ 253 0.12    199/ 205 0.92    179/ 188 0.71    310/ 287 0.20   1945/1948 0.59  seed2
scalar x0.003   120    313/ 253 0.12    199/ 205 0.91    184/ 191 0.71    307/ 282 0.19   1946/1963 0.59  seed4
scalar x0.01     60    313/ 253 0.12    196/ 197 0.93    165/ 181 0.78    196/ 177 0.32   1651/1712 0.71  seed0
scalar x0.01     60    313/ 253 0.12    196/ 199 0.94    164/ 179 0.81    205/ 177 0.32   1677/1746 0.71  seed2
scalar x0.01     60    313/ 253 0.12    194/ 197 0.94    163/ 178 0.77    223/ 188 0.32   1712/1733 0.71  seed4
scalar x0.01    120    313/ 253 0.12    197/ 202 0.91    172/ 180 0.70    304/ 266 0.19   1866/1893 0.57  seed0
scalar x0.01    120    313/ 253 0.12    196/ 203 0.91    171/ 184 0.69    293/ 261 0.19   1828/1862 0.57  seed2
scalar x0.01    120    313/ 253 0.12    197/ 203 0.90    175/ 185 0.69    299/ 284 0.19   1877/1917 0.59  seed4
scalar x0.03     60    313/ 253 0.12    194/ 195 0.94    152/ 171 0.78    180/ 146 0.29   1537/1606 0.69  seed0
scalar x0.03     60    313/ 253 0.12    194/ 198 0.94    151/ 170 0.79    179/ 141 0.28   1572/1609 0.70  seed2
scalar x0.03     60    313/ 253 0.12    195/ 193 0.93    158/ 175 0.78    190/ 158 0.28   1589/1662 0.70  seed4
scalar x0.03    120    313/ 253 0.12    195/ 198 0.91    164/ 177 0.70    288/ 234 0.18   1784/1742 0.55  seed0
scalar x0.03    120    313/ 253 0.12    192/ 199 0.91    162/ 173 0.70    284/ 214 0.17   1704/1692 0.53  seed2
scalar x0.03    120    313/ 253 0.12    193/ 197 0.91    165/ 176 0.69    263/ 234 0.19   1741/1729 0.57  seed4
norm S=0.0003    60    313/ 253 0.12    200/ 202 0.93    175/ 183 0.79    257/ 206 0.31   1841/1806 0.70  seed0
norm S=0.0003    60    313/ 253 0.12    197/ 202 0.93    177/ 184 0.81    260/ 199 0.33   1862/1886 0.71  seed2
norm S=0.0003    60    313/ 253 0.12    198/ 204 0.92    176/ 183 0.78    261/ 200 0.31   1828/1780 0.72  seed4
norm S=0.0003   120    313/ 253 0.12    200/ 204 0.92    182/ 187 0.71    330/ 277 0.21   1969/1951 0.59  seed0
norm S=0.0003   120    313/ 253 0.12    200/ 205 0.93    181/ 189 0.72    328/ 289 0.19   1973/1976 0.58  seed2
norm S=0.0003   120    313/ 253 0.12    200/ 204 0.91    183/ 189 0.71    333/ 300 0.21   1974/1989 0.60  seed4
norm S=0.001     60    313/ 253 0.12    195/ 201 0.94    163/ 176 0.77    246/ 179 0.30   1761/1738 0.70  seed0
norm S=0.001     60    313/ 253 0.12    195/ 200 0.93    163/ 170 0.79    259/ 185 0.30   1782/1724 0.68  seed2
norm S=0.001     60    313/ 253 0.12    195/ 201 0.94    162/ 175 0.76    253/ 216 0.31   1777/1790 0.70  seed4
norm S=0.001    120    313/ 253 0.12    198/ 203 0.91    166/ 179 0.69    322/ 291 0.19   1895/1909 0.57  seed0
norm S=0.001    120    313/ 253 0.12    197/ 203 0.91    166/ 178 0.69    321/ 274 0.20   1872/1926 0.59  seed2
norm S=0.001    120    313/ 253 0.12    199/ 201 0.90    172/ 182 0.69    310/ 292 0.21   1913/1913 0.59  seed4
norm S=0.003     60    313/ 253 0.12    193/ 196 0.94    134/ 151 0.75    222/ 133 0.30   1636/1404 0.58  seed0
norm S=0.003     60    313/ 253 0.12    195/ 198 0.96    137/ 148 0.78    202/ 129 0.30   1547/1347 0.59  seed2
norm S=0.003     60    313/ 253 0.12    193/ 198 0.94    138/ 149 0.75    233/ 182 0.31   1654/1656 0.70  seed4
norm S=0.003    120    313/ 253 0.12    193/ 198 0.91    147/ 158 0.65    293/ 265 0.21   1783/1824 0.57  seed0
norm S=0.003    120    313/ 253 0.12    194/ 199 0.91    143/ 157 0.65    297/ 263 0.19   1809/1803 0.56  seed2
norm S=0.003    120    313/ 253 0.12    196/ 197 0.91    147/ 158 0.62    299/ 257 0.20   1832/1781 0.56  seed4

VERDICT (60 Hz unless stated; ALPN J <= 0.35, KC J <= 0.20, KC act 0.02-0.10 both odours every seed, central 0.01-0.05, 120 Hz central < 0.08)
arm                ALPN J        KC J      KC act     central   120Hz cen  verdict
base                0.942       0.916 0.347-0.351       0.162       0.167  MISS(alpn_j,kc_j,kc_act,central,cen120)
delete              0.842       0.316 0.040-0.053       0.059       0.065  MISS(alpn_j,kc_j,central)
negate              0.203       0.035 0.004-0.026       0.023       0.038  MISS(kc_act)
scalar x0.003       0.825       0.320 0.036-0.049       0.056       0.061  MISS(alpn_j,kc_j,central)
scalar x0.01        0.785       0.320 0.034-0.043       0.053       0.059  MISS(alpn_j,kc_j,central)
scalar x0.03        0.786       0.284 0.027-0.037       0.049       0.055  MISS(alpn_j,kc_j)
norm S=0.0003       0.794       0.318 0.038-0.050       0.057       0.061  MISS(alpn_j,kc_j,central)
norm S=0.001        0.773       0.304 0.035-0.050       0.054       0.059  MISS(alpn_j,kc_j,central)
norm S=0.003        0.760       0.301 0.025-0.045       0.048       0.057  MISS(alpn_j,kc_j)

NO ARM PASSES; best negate fails on kc_act; failure mode dead
```

**Verdict: no arm passes.** Thresholds are the plan Success Metrics unchanged -
ALPN Jaccard <= 0.35, KC Jaccard <= 0.20, KC active 2-10% for BOTH odours on
EVERY seed, central 1-5%, 120 Hz central < 8%. The best arm is the `negate`
bridge, which misses on one metric only (KC occupancy) and whose failure mode is
**dead**: odour B reaches 0.4%, 0.5% and 1.1% of the 5,177 KCs at seeds 0/2/4,
below the 2% floor, while odour A sits in band at 2.3-2.6%. Every electrical arm
fails the opposite way, **merged**: ALPN Jaccard 0.76-0.84 and KC Jaccard
0.28-0.32, three to four times the thresholds, with the brain comfortably alive.

**`negate` reproduces section 7.** All six `negate` rows here are identical to the
s7 rows at 60 and 120 Hz - ALPN J 0.20 / 0.20 / 0.21 at 60 Hz, KC J 0.01 / 0.03 /
0.06, central 948/433, 870/444, 964/893 - well inside the 0.01 tolerance, so the
sweep harness is measuring the same brain the bridge was measured on.

**Marginal rows.** Two arms pass a metric by a hair and should not be read as
comfortable: `scalar x0.03` has central 0.049 against a 0.05 ceiling, and
`norm S=0.003` has central 0.048 and a KC-act minimum of 0.025 against a 0.02
floor. Both are within one seed's scatter of flipping; neither changes the
verdict, since both miss ALPN Jaccard by more than 2x regardless.

**How far the arms were pushed, and by what.** The two coupling arms are bounded
differently, and only one bound is measured:

- The SCALAR arm is bounded above by evidence. Section 8 measured `kmax * 0.1`:
  ALPN Jaccard 0.586 - still 1.7x the threshold - with KC occupancy already at
  1.7%, below the 2% floor. So the scalar axis was walked past the point where
  the KC code dies and separation still had not arrived; 0.03 is the last value
  with a live KC layer, not an arbitrary stopping point.
- The NORMALISED arm's range was set by the PLAN, not by a measured ceiling: the
  physiological band S ~ 0.0005-0.001 and 6x it. Nothing here shows S = 0.003 is
  near any limit. Section 8's clamp is at a summed coefficient of 0.5, i.e. 166x
  above S = 0.003, so most of the normalised axis is simply unmeasured.

**Opposing trends.** Over the 10x normalised range, ALPN Jaccard moves
0.794 -> 0.760 (~0.03 per decade) while the KC-act minimum moves 0.038 -> 0.025,
toward the 0.02 floor. Extrapolating both: KC occupancy leaves the band about a
decade out, whereas ALPN Jaccard at ~0.03 per decade needs of order ten more
decades to reach 0.35. The two metrics move in opposite directions relative to
their bands and KC hits its floor first by a wide margin, which is why extending
the normalised range is not expected to produce a passing arm - though, per the
point above, that is an extrapolation, not a measurement.

**Attribution.** Deleting the chemical eLN broadcast is what moves the regime:
`delete` (the same 3,802 edges zeroed, coupling ~nil) takes ALPN Jaccard 0.942 ->
0.842 and KC occupancy 0.35 -> 0.04-0.05 on its own. Electrical coupling added on
top buys 0.842 -> 0.786 over a 10x scalar range, and per-target normalisation
buys 0.842 -> 0.760 at its strongest measured point, each step costing KC
occupancy - so the primitive as specified adds a slow, monotone trim to what the
deletion already did and does not reach the separation the criterion asks for.
The two strongest arms are roughly strength-matched, so that comparison is fair:
total coupling summed over all 3,802 edges is `f * 55471 * kmax` = 0.85 for
`scalar x0.03` and `337 * S` = 1.01 for `norm S=0.003`, within ~20% of each
other. What differs between them is the DISTRIBUTION, which is the whole point of
`GAP_NORM`, and redistributing it buys 0.786 -> 0.760.

**Consequence for phase 4.** Per the selection rule, with no passing arm the
criterion-2 work proceeds on `negate` (the bridge) with the KC-occupancy miss on
the record. The electrical primitive stays default-off and unchanged; the next
candidate to test is source-side (per-eLN) normalisation, or accepting that the
44's chemical deletion plus the bridge is the working regime.

## 10. Criterion 2 on the bridge: ACCEPT, and vacuous (phase 4, closed)

`learn/condition.py --state v2neg --train 30` with `gpu_sim.ELN_NEGATE = True` (set in a
wrapper before `main()`; no other constant moved; 1,219 s on this GPU). `--check`:

```
learn: misery +6.077 skyhigh -5.354 (baseline sd 0.939); lesion: misery +0.437
ACCEPT
```

Those numbers are the v1 numbers to three decimals. Diffing `condition_v2neg.jsonl`
against `condition_v1.jsonl` row for row (660 rows): `avoid_index`, `kc_active`,
`central_active`, `mbon_avoid_hz` and every other readout key are bit-identical; the
only differing key is `w_n_at_floor` (v1 0, v2neg ~450), which is the counter fixed in
`4968aed` at 13:28, after the v1 run at 13:12, not an effect of the bridge.

**Reading.** The song-conditioning paradigm never traverses the 44 eLNs' edges. Its
drive enters through the auditory path and reaches the mushroom body without the
antennal lobe, so negating the eLNs cannot change a single spike in it, and it did not.
Criterion 2 as written in the plan ("`condition.py --check` prints ACCEPT on that
brain") is therefore satisfied and says nothing about whether odour separation
survives plasticity. A criterion 2 that would mean something needs an odour-conditioned
paradigm (two multi-glomerular odours as CS+, CS-) driven through the AL; that is a new
spec, not a rerun.

**Decision (2026-09-20).** Stopped here. The plan's hypothesis (electrical eLN -> PN
coupling reaches criterion 1) is refuted over the range measured (s9); the bridge
separates odours but leaves odour B at 0.4-2.6% KC occupancy (s7); criterion 2 could
not be tested by the existing paradigm (this section). What stands: the sim's
one-attractor regime is caused by the 44 eLNs' chemical broadcast (s2-s6); deleting it
does most of the work (s8-s9); `ELN_NEGATE` is the only measured separating regime and
is labelled a hack; `GAP_COUPLE`/`GAP_NORM` remain in the code, default off, tested,
for whoever revisits with source-side normalisation or an odour-conditioning paradigm.

**Addendum, chess drive.** `drive_of(chess.Board())` stimulates 3,783 neurons, 725 of them
olfactory (plus 3,058 optic). Start position, seed 0: OFF `central>1Hz` 0.172, KC 37.7%
active (the ignited attractor); `ELN_NEGATE` 0.047 and 4.4%. The chess reservoir has
been running inside the AL ignition the whole time, and the bridge takes it out. Not
measured: whether board-space locality (where_locality_dies.py, 2026-09-16: r = 0.08 at
the rates stage) survives better on the negated brain. That is the next cheap
experiment if the chess side is revisited.

## Next

1. ~~Decide the sign question~~ Done, section 6: real eLNs, dynamics problem. Spec
   candidate: electrical coupling for eLN -> PN (graded, bidirectional, sub-threshold)
   instead of chemical delivery; the other 90 low-confidence excitatory ALLNs stay an open
   data question.
2. ~~Decide the sign question honestly~~ (superseded by 1): pull the FlyWire nt prediction confidence for the
   134 (cell ids are in `neuron_meta.npz` order; the annotation TSV is Schlegel et al.
   2024) before treating the flip as a correction rather than a hack. If confidence is
   low, the flip is a data correction; if high, the real eLNs exist and the fix belongs
   in the dynamics (glomerulus-local eLN gain), which needs a spec.
2. Make multi-glomerular odour pairs the standard probe (`regime/alln_multiglom.py`
   seeds them); retire single-glomerulus KC Jaccard as a criterion.
3. Then the criterion-1 arms (APL, SFA) on top, and criterion 2 (plasticity).
