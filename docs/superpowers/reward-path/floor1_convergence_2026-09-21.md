# Brain-wide floor-1: sugar -> rewarding PAM is 30 synapses at its widest. Reward through anatomy is CLOSED (2026-09-21)

**[2026-09-23 correction: a synapse fraction is not a firing test. Christie et al. 2026 (PMC12869359) report PAM firing via Fox/FDA in a Shiu LIF, and the path IS in v783 to the synapse (Fox -> CB0233 = 217, Fox -> FDA-I = 171). Measured verdict, with silencing controls: the chain conducts to FDA and dies at FDA-I -> PAM (0/307 PAMs fire). See christie_reconciliation_2026-09-23.md - cite that, not this fraction.]**

Track 1 of `fly/.claude/PRPs/plans/neuroplas-roadmap-2026-09-21.md`. The last open
lever from `signed_paths_2026-09-21.md`: the 53-synapse result was measured on our
patched npz, which thresholds edges. Maybe the taste channel is real and we cut it.

**It is not.** On the raw feather at floor 1 - every proofread edge in v783, nothing
thresholded, 15,090,883 pre/post pairs, 54,490,417 synapses - the widest sugar-driven
channel onto a literature reward subtype carries **30 synapses**. Falsifier threshold
was 100. Closed.

## Method

`regime/sugar_pam_floor1.py`, ~4 min CPU, read-only, no npz and no sim.

- Edges: `data/ext/proofread_connections_783.feather`, per-neuropil rows summed per
  (pre, post), floor 1. 16,846,880 of 16,847,997 rows (99.99%) map into our
  139,248-neuron index; the rest are endpoints outside the proofread set.
- Seed: the same 129 `cell_sub_class == "sugar/water"` GRNs as every earlier probe.
- Targets: `hemibrain_type` prefixes PAM01..PAM15, joined from the annotations tsv
  (`neuron_meta.npz` still has no `hemibrain_type`). All 15 reported, so the choice of
  PAM02 / PAM04 / PAM11 (beta'2a, gamma4, gamma3) as "the rewarding ones" cannot hide a hit.
- Reach: forward, >= 1 synapse per hop. Hop-1 720 neurons, hop-2 20,428, hop-3 116,964.

### The metric that matters is the bottleneck, not the raw sum

A hop-2 carrier can only relay what sugar delivers *to* it. Raw hop-2 synapse mass
overstates the channel every time a neuron with a strong PAM projection happens to pick
up one or two sugar synapses. The clearest case: **CB0032 receives 1 synapse from the
sugar GRNs and sends 112 onto PAM11** - that single neuron is 68% of PAM11's raw hop-2
total. So the reported number is `sum over carriers of min(sugar -> carrier, carrier -> subtype)`.

## Result

| subtype | n | total in-syn | hop2 carriers | raw hop2 syn | **bottleneck** | carriers w/ sugar_in>=5 | their syn | hop3 carriers | hop3 syn |
|---|---|---|---|---|---|---|---|---|---|
| PAM01 | 41 | 11407 | 9 | 118 | **34** | 5 | 70 | 1077 | 4697 |
| PAM02 | 16 | 6172 | 8 | 16 | **13** | 3 | 7 | 534 | 2195 |
| PAM03 | 10 | 1913 | 1 | 5 | **5** | 1 | 5 | 240 | 558 |
| PAM04 | 32 | 16600 | 14 | 55 | **20** | 4 | 6 | 1123 | 5058 |
| PAM05 | 21 | 10426 | 6 | 14 | **6** | 2 | 2 | 690 | 4157 |
| PAM06 | 30 | 16170 | 5 | 34 | **19** | 2 | 27 | 595 | 5324 |
| PAM07 | 18 | 5650 | 1 | 1 | **1** | 1 | 1 | 485 | 1697 |
| PAM08 | 45 | 16522 | 7 | 12 | **11** | 4 | 6 | 1048 | 5706 |
| PAM09 | 9 | 5588 | 3 | 7 | **6** | 2 | 3 | 487 | 1309 |
| PAM10 | 15 | 15316 | 9 | 65 | **21** | 2 | 17 | 1035 | 4736 |
| PAM11 | 16 | 8813 | 12 | 164 | **30** | 4 | 16 | 891 | 3167 |
| PAM12 | 23 | 4436 | 2 | 35 | **4** | 1 | 1 | 511 | 1590 |
| PAM13 | 12 | 3070 | 0 | 0 | **0** | 0 | 0 | 348 | 1236 |
| PAM14 | 16 | 3425 | 3 | 5 | **4** | 0 | 0 | 317 | 1197 |
| PAM15 | 3 | 1513 | 4 | 9 | **9** | 3 | 8 | 238 | 489 |
| **ALL PAM** | **307** | **127021** | **38** | **540** | **115** | **11** | **169** | 3552 | 43116 |

Raw JSON: `results/sugar_pam_floor1.json` (gitignored; numbers reproduced above).

### Reading it

- **53 -> 540 raw hop-2 synapses.** Floor 1 does widen the channel 10x over the patched
  npz, and the widening is on the *sugar -> carrier* edges, not carrier -> PAM: the
  patched matrix found 221 hop-1 neurons, floor 1 finds 720. So our threshold was hiding
  edges. They are weak edges.
- **540 raw is still 0.43% of PAM's 127,021 input synapses**, and the bottleneck-corrected
  figure is **115 across all 307 PAM**, 0.09%.
- **Per rewarding subtype: PAM02 13, PAM04 20, PAM11 30.** The falsifier asked for < ~100
  per subtype. All three are under a third of it. The competing branch asked for
  *thousands*; nothing in the table is within two orders of magnitude.
- **The carriers are the same five neurons plus noise.** CB0233, CB0643, CB0130, CB0687
  from the 53-synapse result, joined at floor 1 by SLP234, CB0032, CB0546, CB3199, a few
  mAL and two ALPNs - almost all of which receive 1-3 synapses from sugar. Only 11 of the
  38 hop-2 carriers onto any PAM receive >= 5 sugar synapses at all.
- **Hop-3 is not a channel, it is the brain.** 116,964 of 139,248 neurons are 3 hops from
  the sugar GRNs. The 43,116 hop-3 synapses onto PAM are generic hubs and Kenyon cells,
  the same finding as the earlier signed-path hop-3 table. No taste specificity to read.

## Verdict

**Reward-through-anatomy is closed.** The sugar -> PAM path is absent from v783
connectivity as reconstructed. **[2026-09-23 correction: a synapse fraction is not a firing test. Christie et al. 2026 (PMC12869359) report PAM firing via Fox/FDA in a Shiu LIF, and the path IS in v783 to the synapse (Fox -> CB0233 = 217, Fox -> FDA-I = 171). Measured verdict, with silencing controls: the chain conducts to FDA and dies at FDA-I -> PAM (0/307 PAMs fire). See christie_reconciliation_2026-09-23.md - cite that, not this fraction.]** This is not our build: the edges are all there at floor 1
and the channel is still 0.09% of PAM input, with no neuron receiving both substantial
sugar drive and projecting substantially onto a reward subtype. A LIF has nothing to
integrate, and no gain, floor, gate or drive rate fixes a missing wire - the levers
already measured out (eta, ORN/GRN drive rate, APL gain, KC threshold, synapse floor,
in-degree, restored-input sign, normalisation, hunger gating) were all downstream of this.

**Consequence for the model:** the US stays injected at the DANs, disclosed as such.
Standing disclosure line: *"the connectome as published does not carry taste to the reward
neurons, so the unconditioned stimulus is injected at the dopaminergic neurons directly."*

**Consequence for the science:** this is a publishable connectome-gap claim with a control
story, in the same family as the APL and PAM-floor precedents - the pathway is
well-established in vivo (Krashes 2009, Huetteroth 2015, Tsao 2018) and measurably not in
the reconstruction. The positive control is already in hand: the same 150 Hz sugar drive
reproduces Shiu et al.'s proboscis motor-neuron response (37.2 Hz from silence), so the
drive works and the specific taste -> reward wire is what is missing.

**What is NOT claimed:** that the pathway does not exist in the animal, or that v783 is
wrong. Candidate explanations not tested here: the carriers are multi-hop through
neuropil our index drops, the relevant GRNs are outside `cell_sub_class == "sugar/water"`,
or reward reaches PAM through neuromodulation / peptidergic signalling that a synaptic
connectome does not carry at all. The last is the most likely and is not falsifiable
with this data.

## Effect on the roadmap

- `reward-path.prd.md` phases 4 and 5: **closed, will not run.** Phase 4 needed a working
  `--us-path grn`; there is no grn path to use.
- Punishment (PPL1) is unaffected: 9/10 seeds, and every learning result to date is
  punishment-only. Rung A and rung 3 (persistence / interference) do not need reward.
- Next live rung is Track 2: odour-CS conditioning on DA1 vs DC2 in the measured regime.

## Reproduce

```
.venv/Scripts/python.exe regime/sugar_pam_floor1.py
```

Needs `data/ext/proofread_connections_783.feather` (852 MB) and the annotations tsv.
Peak RSS ~6 GB.
