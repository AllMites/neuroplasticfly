# Signed paths sugar GRN -> PAM: the restored input is excitatory, and there is almost none of it (2026-09-21)

Follow-up to `dan_floor_2026-09-21.md`, which measured out input in-degree, the
synapse floor and GRN drive strength as the reward fix. Five levers were run
here, each with a falsifier fixed in advance. **Nothing was tuned.** Four of the
five came back negative for their own hypothesis; the one positive result is
structural and it kills the "net inhibition" story outright.

Scripts: `regime/signed_paths.py` (L1 + the L3 variant npz),
`learn/pam_intermediates_sweep.py` (L2 / L3 / L4 runs).
Raw: `results/signed_paths.json`, `results/pam_intermediates_{base3,danfloor1,nokc,danfloor1_norm}.json`.
Matrices: `data/brain_gpu.npz` (2,700,429 edges), `data/brain_gpu_danfloor1.npz`
(2,804,652, `dan_floor` 1), **new** `data/brain_gpu_danfloor1_nokc.npz`
(2,743,080, patched minus the restored KC->PAM edges).

Command lines:

```
.venv/Scripts/python.exe regime/signed_paths.py --write-nokc
.venv/Scripts/python.exe learn/pam_intermediates_sweep.py --seeds 3 --tag base3
FLYCHESS_BRAIN=data/brain_gpu_danfloor1.npz      .venv/Scripts/python.exe learn/pam_intermediates_sweep.py --seeds 3 --tag danfloor1
FLYCHESS_BRAIN=data/brain_gpu_danfloor1_nokc.npz .venv/Scripts/python.exe learn/pam_intermediates_sweep.py --seeds 3 --tag nokc
FLYCHESS_BRAIN=data/brain_gpu_danfloor1.npz      .venv/Scripts/python.exe learn/pam_intermediates_sweep.py --seeds 3 --tag danfloor1_norm --norm 29.15
```

Matrix selection is `FLYCHESS_BRAIN`, read at import by `gpu_sim`. `--norm` sets
`gpu_sim.NORM_TOTAL_TARGET` **before** `GpuSim()` (it is read once in
`__init__`); every other run in this doc has it at 0.0, as in the dan_floor doc.

---

## L1. Signed-path analysis, sugar GRN -> PAM, patched matrix (structural)

### 1a. The restored PAM input is mostly excitatory

"Restored" here = edges present in `brain_gpu_danfloor1.npz` and absent from
`brain_gpu.npz`, postsynaptic to PAM. **104,223** edges are new brain-wide
(82,144 onto PAM); the dan_floor doc's 109,603 counts all target edges written,
including the 5,380 pre-existing ones that were removed and rewritten
identically. Sign is the presynaptic `top_nt` per `NT_SIGN` in
`patch_dan_floor.py` (gaba/glutamate -1, everything else incl. `none` +1).

| restored PAM input | pairs | synapses | inhibitory synapses | frac inhib |
|---|---|---|---|---|
| all | 82,144 | 111,167 | 11,892 | **0.107** |
| KC -> PAM | 61,572 | 80,505 | 16 | 0.0002 |
| non-KC -> PAM | 20,572 | 30,662 | 11,876 | **0.387** |

Non-KC restored input by presynaptic transmitter:

| presyn top_nt | sign | pairs | syn |
|---|---|---|---|
| acetylcholine | +1 | 10,607 | 15,906 |
| glutamate | -1 | 5,418 | 8,278 |
| gaba | -1 | 2,328 | 3,596 |
| dopamine | +1 | 2,087 | 2,697 |
| serotonin | +1 | 82 | 132 |
| octopamine | +1 | 50 | 53 |

Per PAM neuron the patch restored 362.1 synapses on average (max 1,184, zero
PAMs missed); the non-KC part is 99.9 mean, 287 max, again no PAM at zero.

### 1b. The signed path products are net excitatory

Walks of exactly k steps from the 129 sugar GRNs to the 307 PAM, weighted by the
product of synapse counts, signed by the product of edge signs. Excitatory and
inhibitory path mass are `(abs +/- signed)/2`.

| | signed net | abs total | frac inhibitory |
|---|---|---|---|
| hop 1 | 0 | 0 | - |
| hop 2 | **+577** | 619 | 0.034 |
| hop 3 | **+2.598e6** | 5.509e6 | 0.264 |

### 1c. But the hop-2 channel is five neurons and 53 synapses

Sugar GRNs reach 221 neurons in one hop. Exactly **five** of those touch a PAM
at all:

| hop-1 neuron | cell_type | sugar -> it (syn) | it -> PAM (syn) | signed | onto n PAM |
|---|---|---|---|---|---|
| 17184 | CB0643 | 5 | 21 | +21 | 6 |
| 111382 | CB0687 | 7 | 3 | **-3** | 3 |
| 123910 | CB0130 | 19 | 7 | +7 | 4 |
| 126833 | CB0130 | 10 | 8 | +8 | 4 |
| 138357 | CB0233 | 20 | 14 | +14 | 11 |

Total: **53 synapses onto PAM**, against 127,021 PAM input synapses on the
patched matrix - **0.04%**. The strongest carrier (CB0233) spreads 14 synapses
over 11 PAM neurons, i.e. ~1.3 synapses each.

Hop-3 is dominated by Kenyon cells and by generic hubs, not by a taste channel.
Top last-hop contributors by |contribution| (637 neurons contribute):

| cell_class | cell_type | hemibrain_type | n | signed | abs | frac inhib |
|---|---|---|---|---|---|---|
| none | CB0272 | | 2 | +7.56e5 | 1.005e6 | 0.124 |
| none | CB0032 | | 2 | +7.41e5 | 9.76e5 | 0.120 |
| Kenyon_Cell | KCg-m | KCg-m | 181 | +4.71e5 | 5.99e5 | 0.107 |
| none | CB0337 | | 2 | +1.85e5 | 4.21e5 | 0.280 |
| none | CB0631 | hb546771021,... | 2 | +2.68e5 | 3.29e5 | 0.093 |
| Kenyon_Cell | KCab | KCab-s | 32 | +1.77e5 | 2.30e5 | 0.114 |
| ALPN | M_l2PNl20 | M_l2PNl20 | 2 | **-1.29e5** | 1.80e5 | 0.859 |
| none | SMP109 | SMP109 | 1 | +1.24e5 | 1.24e5 | 0.000 |
| MBIN | APL | APL | 1 | **-1.05e5** | 1.05e5 | 1.000 |
| LHCENT | LHCENT3 | LHCENT3 | 1 | **-1.04e5** | 1.04e5 | 1.000 |
| none | OA-VPM4 | OA-VPM4 | 2 | -5.30e4 | 9.27e4 | 0.786 |
| none | CRE011 | CRE011 | 1 | +8.26e4 | 8.26e4 | 0.000 |
| MBON | MBON35 | MBON35 | 1 | +8.13e4 | 8.13e4 | 0.000 |

The plan's guessed US carriers exist: 4,557 non-KC neurons gained a restored edge
onto PAM, 3,616 of them with no `cell_class`/`cell_type` label (the "none"-class
SMP/CRE population). They are numerous; they are just not on the sugar path at
hop 2.

### Verdict L1

**Falsifier fired: "restored PAM input is net inhibitory" is DEAD.** The sugar
-> PAM path is net excitatory by sign product at both hop 2 (+577 of 619, 3.4%
inhibitory) and hop 3 (+2.60e6 of 5.51e6, 26% inhibitory), and the restored PAM
input is 89.3% excitatory by synapse count (61.3% even within the non-KC
subset). The real structural fact is quantitative, not signed: the entire
taste-specific channel onto PAM is 5 neurons and 53 synapses, 0.04% of PAM
input. "306/307 PAM within 3 hops" was reach through KC and through brain-wide
hubs, not through a gustatory pathway.

---

## L2. Dynamic check of the intermediates (3 seeds, patched matrix)

`n_edges` 2,804,652, `dan_floor` 1, `NORM_TOTAL_TARGET` 0.0, seeds 0-2. Mean is
reported next to n, the count below 1 Hz, and the max, because a mean over a
silent population reads far better than the population behaves.

| group | n | base mean | sugar mean | delta | sugar max | base silent | sugar silent |
|---|---|---|---|---|---|---|---|
| sugar GRN (drive check) | 129 | 0.000 | 149.466 | +149.5 | 223.3 | 129/129 | 0/129 |
| proboscis MN (control) | 24 | 0.000 | **37.176** | +37.2 | 143.3 | 24/24 | 11/24 |
| hop2 intermediates (all 5) | 5 | 11.111 | 14.000 | +2.9 | 53.3 | 4/5 | 3/5 |
| hop2 type CB0233 | 2 | 0.000 | **8.889** | +8.9 | 23.3 | 2/2 | 1/2 |
| hop2 type CB0687 (inhibitory) | 2 | 38.889 | 38.333 | -0.6 | 53.3 | 0/2 | 0/2 |
| hop2 type CB0130 | 2 | 0.000 | 0.000 | 0.0 | 0.0 | 2/2 | 2/2 |
| hop2 type CB0643 | 4 | 0.000 | 0.000 | 0.0 | 0.0 | 4/4 | 4/4 |
| hop3 type CB0032 | 2 | 0.000 | 2.778 | +2.8 | 6.7 | 2/2 | 1/2 |
| hop3 type CB0272 | 2 | 0.000 | 1.111 | +1.1 | 3.3 | 2/2 | 1.3/2 |
| hop3 type APL | 2 | 248.889 | 251.667 | +2.8 | 263.3 | 0/2 | 0/2 |
| hop3 type CRE011 | 2 | 53.333 | 50.556 | -2.8 | 96.7 | 0.7/2 | 0.7/2 |
| hop3 types SMP109 / LHCENT3 / OA-VPM4 / MBON35 / MBON24 / LAL119 / M_l2PNl20 / CB0337 / CB0631 / CB0546 / VESa2_P01 | 2 each | 0.000 | 0.000 | 0.0 | 0.0 | all silent | all silent |
| PAM presyn unlabelled (restored, non-KC) | 3,616 | 0.789 | 0.769 | **-0.020** | 96.7 | 3462.7 | 3451.3 |
| **PAM (all)** | 307 | 0.141 | **0.123** | **-0.018** | 13.3 | **302.0/307** | **302.3/307** |

PAM by hemibrain subtype, sugar vs baseline - no subtype is driven, and the mean
is not hiding one:

| subtype | n | base | sugar | subtype | n | base | sugar |
|---|---|---|---|---|---|---|---|
| PAM01 | 41 | 0.000 | 0.000 | PAM09 | 9 | 0.000 | 0.000 |
| PAM02 | 16 | 0.000 | 0.000 | PAM10 | 15 | 0.000 | 0.000 |
| PAM03 | 10 | 0.000 | 0.000 | PAM11 | 16 | 0.000 | 0.000 |
| PAM04 | 32 | 0.000 | 0.000 | PAM12 | 23 | 0.000 | 0.000 |
| PAM05 | 21 | 0.529 | 0.423 | PAM13 | 12 | 0.000 | 0.000 |
| PAM06 | 30 | 0.407 | 0.407 | PAM14 | 16 | 0.000 | 0.000 |
| PAM07 | 18 | 0.000 | 0.000 | PAM15 | 3 | 0.000 | 0.000 |
| PAM08 | 45 | 0.444 | 0.370 | | | | |

Nine of fifteen subtypes are **completely silent in both conditions**. The five
non-silent PAM neurons sit in PAM05/06/08 and all move *down* or flat under
sugar. Regime: kc_active 0.0500 both (authored, invariant), central_active
0.0249 -> 0.0344.

### Verdict L2

**Falsifier outcome: the block is at the last synapse, not upstream.** The
intermediates the structural analysis named *are* driven - CB0233 goes 0 ->
8.889 Hz, CB0032 0 -> 2.778, CB0272 0 -> 1.111, and the proboscis MN control
reaches 37.2 Hz from silence - while PAM stays flat and 302/307 silent. It is
not inhibition at that synapse either: CB0233's 14 synapses onto 11 PAM are
excitatory. **It is fan-out.** ~1.3 excitatory synapses per PAM from a neuron
firing 9 Hz is orders of magnitude below what a LIF PAM needs. The two
substantial inhibitory hop-3 hubs that *are* active (APL 249 Hz, and CB0687 at
38 Hz inhibitory at hop 2) are plausibly holding PAM down, which explains the
0.018 Hz *drop*, but the drop is noise-sized; there is no excitatory drive for
them to be cancelling.

---

## L3. KC -> PAM dominance

Structural (from `results/signed_paths.json`): of 111,167 restored PAM input
synapses, **80,505 (72.4%) are KC -> PAM** across 61,572 pairs; the non-KC
remainder is 30,662 synapses over 20,572 pairs. Per PAM: 362.1 restored synapses
mean, of which only 99.9 non-KC.

`data/brain_gpu_danfloor1_nokc.npz` is the patched matrix with exactly those
61,572 restored KC->PAM edges dropped (nnz 2,804,652 -> 2,743,080). KC->PAM
edges that already survived the >=5 floor are kept - only the restored ones go.

3-seed comparison, same seeds, `NORM_TOTAL_TARGET` 0.0:

| matrix | n_edges | PAM base | PAM sugar | PAM silent (base) | PAM sugar max | central_active base |
|---|---|---|---|---|---|---|
| `brain_gpu.npz` | 2,700,429 | 0.025 | 0.033 | 305.7/307 | 6.67 | 0.0193 |
| `brain_gpu_danfloor1.npz` | 2,804,652 | **0.141** | 0.123 | 302.0/307 | 13.33 | 0.0249 |
| `brain_gpu_danfloor1_nokc.npz` | 2,743,080 | **0.112** | 0.094 | 303.3/307 | 13.33 | 0.0249 |

### Verdict L3

**Falsifier outcome: NOT KC noise.** Stripping every restored KC->PAM edge moves
the PAM baseline 0.141 -> 0.112 Hz, i.e. it accounts for ~25% of the patch's
baseline rise (0.025 -> 0.141), not all of it. The other ~75% comes from the
restored non-KC input. Sugar still does nothing without KC->PAM (0.112 ->
0.094, again slightly *down*), so the reward failure is unchanged by this
variant and the npz has no further use except as a control.

---

## L4. Normalisation arm (plan risk row, never previously run)

`NORM_TOTAL_TARGET = 29.15`, the value `regime/measure.py` uses, on the patched
matrix, 3 seeds. **This is a measurement, not a fix.**

| group | n | base | sugar | silent (sugar) |
|---|---|---|---|---|
| sugar GRN (drive check) | 129 | 0.000 | 149.466 | 0/129 |
| proboscis MN (control) | 24 | 0.000 | **0.000** | 24/24 |
| hop2 intermediates | 5 | 0.000 | 0.000 | 5/5 |
| PAM (all) | 307 | 0.000 | 0.000 | **307/307** |
| every PAM subtype PAM01-15 | - | 0.000 | 0.000 | all |

central_active 0.0087 (base) / 0.0092 (sugar), against 0.0249 / 0.0344 unnormalised.

### Verdict L4

**Falsifier fired: normalisation nulls the effect - and everything else with
it.** Under `NORM_TOTAL_TARGET = 29.15` the drive does not leave the GRNs: even
the proboscis MN positive control, 37.2 Hz unnormalised, goes to **0.000 Hz**,
and central_active falls to a third. The per-postsynaptic factor divides out the
in-degree the patch restored, and on top of that this constant was tuned in
`regime/measure.py` alongside `SFA_B_INC` and `APL_GRADED`; applied alone it
produces a near-silent brain. So this arm reports nothing about reward
specifically. **No number from a normalised run is comparable to any number in
this doc or in `dan_floor_2026-09-21.md`.** The plan risk row "normalisation nulls
the added drive" is answered: yes, catastrophically, which also means it can
never be the reward fix.

---

## L5. Literature check: is the drive physiological, and is PAM state-gated?

**150 Hz sugar GRN drive is inside Shiu et al.'s own protocol, not above it.**
Shiu et al. 2024 (*Nature* 634:210-219) scanned predicted downstream firing for
sugar GRN drive across **10-200 Hz**, and activated candidate seed sets at 50,
100, 150 and 200 Hz; baseline firing for every neuron in their model is 0 Hz, so
the "drive" is an optogenetic-style clamp rather than a claim about in-vivo
GRN spike rates. Our 150 Hz sits mid-range of what they published, and it
reproduces their sugar -> proboscis result (37.2 Hz on proboscis MNs from
silence). **PAM activation by sugar is hunger-gated in vivo, and by a mechanism
the connectome alone cannot express.** Krashes et al. 2009 (*Cell*
139:416-427) showed appetitive-memory expression is suppressed by satiety and
restored by dNPF/NPY-ortholog signalling onto MB-projecting dopaminergic
neurons - a neuropeptide gate, not a synaptic one. Huetteroth et al. 2015
(*Curr Biol* 25:751-758) split the rewarding PAM population by sweet taste vs
nutrient value and found that direct stimulation of the nutrient-reinforcing
PAM subsets **cannot implant appetitive long-term memory in food-satiated
flies**, i.e. hunger establishes a permissive internal state. Tsao et al. 2018
(*eLife* 7:e35264) showed starvation bidirectionally modulates MBON responses to
food odour through six DAN types, and Yamagata et al. 2015 (*PNAS*
112:578-583) and Liu et al. 2012 (*Nature* 488:512-516) established that PAM
subsets are the sugar-reward carriers whose activation "is increased on
starvation", with PAM-gamma3 a target of the inhibitory satiety peptide AstA.
**Conclusion: "state gating" is a credible next plan** - but note it predicts a
*modulation* of an existing drive, and L1/L2 show there is no drive here to
modulate (53 synapses, 0.04% of PAM input). A hunger model on this matrix would
gate a channel that is structurally absent.

---

## Verdict table

| lever | hypothesis | falsifier outcome |
|---|---|---|
| L1 | restored PAM input is net inhibitory | **DEAD** - path net excitatory at hop 2 and 3; restored input 89.3% excitatory |
| L2 | block is upstream (intermediates not driven) | **DEAD** - intermediates are driven (CB0233 0 -> 8.9 Hz); block is at the last synapse, by fan-out not by sign |
| L3 | patch's baseline rise is KC->PAM noise | **DEAD** - stripping restored KC->PAM gives 0.141 -> 0.112, ~25% of the rise; reward unchanged |
| L4 | normalisation nulls or flips the effect | **CONFIRMED and uninformative** - it nulls the whole brain, MN control 37.2 -> 0.000 Hz |
| L5 | 150 Hz unphysiological / PAM not state-gated | 150 Hz is inside Shiu's 10-200 Hz scan; PAM sugar reward **is** hunger-gated in vivo (Krashes 2009, Huetteroth 2015, Tsao 2018) |

## How to apply

**Now measured out. Do not re-propose any of these as the reward fix:**

- input in-degree / the >=5 synapse floor (dan_floor doc)
- GRN drive rate, eta, APL gain (dan_floor doc)
- net inhibition of the restored PAM input (L1)
- an upstream block, i.e. intermediates being under-driven (L2)
- KC->PAM restored edges as the explanation of the patch's PAM baseline (L3)
- `NORM_TOTAL_TARGET` as a knob that could rescue reward (L4) - it silences the
  positive control, so it is disqualified, not merely negative
- "150 Hz is too hot" (L5)

**The one number that now explains the failure:** the sugar-specific excitatory
channel onto PAM is 5 neurons and **53 synapses**, 0.04% of PAM's 127,021 input
synapses on the patched matrix, spread ~1.3 synapses per PAM. There is no
convergence for a LIF to integrate. This is a *sparsity* result, and unlike the
in-degree result it does not go away by restoring edges - the edges are already
all there at floor 1.

**Next live lever (open, not run).** *Is the sugar -> PAM channel missing from
v783, or is it real and carried by many-synapse convergence we are cutting
elsewhere?* Concretely: take the sugar-responsive PAM subsets named in the
literature (PAM-beta'2a / PAM-gamma4 / PAM-gamma3, via `hemibrain_type`
PAM02/PAM04/PAM11 etc.) and, on the **raw feather at floor 1, brain-wide** (not
just DAN rows), count their shortest-path convergence from the 129 sugar GRNs:
how many distinct hop-2 and hop-3 neurons carry >= 10 synapses onto them, and
what total presynaptic synapse mass a sugar-driven population delivers.
**Falsifier:** if a brain-wide floor-1 matrix still gives < ~100 sugar-driven
synapses per rewarding PAM subtype, then the sugar -> PAM path is absent from
v783 connectivity as reconstructed, the fault is the connectome or its
annotation and not our build, and reward through anatomy is closed - inject at
the DANs and disclose it. If it gives thousands, the DAN-only patch was too
narrow and the global floor-1 matrix (currently in "What We're NOT Building")
becomes the experiment, with the ignition regime as the cost.

Do **not** run a hunger/state-gating plan before that count. State gating
multiplies an existing signal; L1/L2 show the signal is ~0.04% of PAM input, and
a gate on zero is still zero.

## Pipeline notes / gaps hit

- `data/neuron_meta.npz` has no `hemibrain_type` column, so PAM01..PAM15 cannot
  be resolved from it. `regime/signed_paths.py` and
  `learn/pam_intermediates_sweep.py` both join it from
  `data/ext/Supplemental_file1_neuron_annotations.tsv` on `root_id` via
  `data/root_ids_sorted.npy`. If subtype work continues, add the column to the
  meta npz rather than re-parsing a 139k-row tsv per script.
- The dan_floor doc's "109,603 added" is target edges *written*, not edges
  *new*: 5,380 of them already existed at floor 5 and were rewritten with the
  same value. Net-new is 104,223. Both are correct; they answer different
  questions, and a diff of the two npz files gives the smaller one.
- `NORM_TOTAL_TARGET` is a module global read once in `GpuSim.__init__`, so it
  must be set before construction. `learn/pam_intermediates_sweep.py` asserts
  it took effect after the constructor.
- `learn/plastic.py`'s offsets cache is keyed by nnz, so
  `brain_gpu_danfloor1_nokc.npz` (2,743,080) gets its own cache file and cannot
  cross-load with the other two matrices. No action needed; noted because it is
  the thing that would silently corrupt a third-matrix run.
