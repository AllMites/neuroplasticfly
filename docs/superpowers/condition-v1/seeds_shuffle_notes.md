# condition v1: seeds, shuffle control, population DN readout (2026-09-20)

Why: competitor read (`fly/.claude/PRPs/reports/competitor-repos-2026-09-20.md`) showed no
whole-brain fly emulation has dopamine-lesion + pairing-shuffle + multi-seed together, and that
our DNg29 readout was a single hand-picked DN. This run adds all three. Raw output:
`seeds_shuffle_analysis_2026-09-20.txt` (this dir). Artifacts (gitignored):
`results/condition_v1s{0..4}.{json,jsonl,png}`, `results/rates_v1s{0..4}/`, `brain_state/v1s{0..4}/`.

## Protocol
`learn/condition.py --state v1s<N> --seed0 <N> --arms learn,lesion,shuffle[,reversed] --us-path dan --dump`
(`results/batch_seeds.sh`). Seed 0 = v1 seeds; v1s0 learn reproduces v1 bit-exact.
Same eta 1e-3, lam 0.01, W_MIN 0.1, 30 trials, 5 probes, 18,674 plastic KC->MBON edges.

New arm **shuffle**: identical wiring, US, rule and seeds, but the eligibility trace is read
from a permuted KC for every plastic edge (`np.random.default_rng(seed0+12345).permutation(kc_of_edge)`),
so the depression lands on random KC->MBON synapses instead of the song-active ones.
Tests pairing specificity (the "shuffled" arm of flytris / fly-hero).

`--dump` saves every probe block's full rate vector (float16) so all 474 DN cell types can be
read, not five hand-picked ones.

## avoid_index shift, last probe minus baseline, Hz (n = 5 seeds, mean +- sd)

| arm | misery (punished) | skyhigh (rewarded) | mozart (never trained) |
|---|---|---|---|
| learn | **+5.86 +- 0.17** | **-4.91 +- 0.41** | +5.66 +- 0.13 |
| lesion (DAN rule off, US still given) | +0.11 +- 0.29 | +0.02 +- 0.24 | +0.03 +- 0.26 |
| shuffle (random KC pairing) | +0.61 +- 0.48 | -0.01 +- 0.35 | +0.28 +- 0.47 |
| reversed (seed 0 only) | +0.12 | +3.54 | -1.89 |

Learn is 20 sd from lesion on misery and 12 sd on skyhigh. Shuffle retains ~10% of the misery
shift and none of the skyhigh shift at the avoid_index level. Mozart generalisation is stable
across seeds (+5.66 +- 0.13), still larger than the skyhigh shift; unchanged caveat.

## Population descending-neuron readout

Per DN cell type, dS = (trained - baseline) on misery minus the same on skyhigh, from the 5-probe
means. Probe seeds differ between baseline and trial 30, so all arms of one seed share the same
seed noise; the lesion arm is that noise alone. E = dS(arm) - dS(lesion) is the plasticity effect.

| statistic | value |
|---|---|
| r(E_learn) between seeds, 10 pairs | +0.976 to +0.997 |
| r(E_reversed, E_learn), seed 0 | **-0.907** |
| r(E_shuffle, E_learn), 5 seeds | +0.64 to +0.98, magnitude 0.21/0.71 = ~30% |
| DN types with \|mean E\| > 3 sd across seeds | **52 / 474** |
| DN types with \|E\| > 1 Hz, learn | 59-62 per seed |
| neurons touched by learning at all (learn vs lesion, trial-30 probe) | 5.9% of DNs, 1.6% of central, 0% optic / sensory / VPN |

Top DN types by mean E (Hz, negative = more suppressed after misery than after skyhigh):
DNge053 -29.3 +- 1.4, DNbe007 -28.5 +- 1.3, DNge037 -18.1 +- 1.4, DNg100 -15.4 +- 2.2,
DNg13 -13.8 +- 1.4, DNp31 -12.5 +- 0.8, DNa02 -12.0 +- 1.4, DNg111 -9.7, DNg16 -9.3, DNb05 -9.1.

Reading: reversing the contingency reverses the whole DN output profile (r -0.91). Random
pairing keeps the direction at ~30% magnitude: that fraction is nonspecific approach-MBON
depression (the same mechanism that makes Mozart generalise); ~70% requires the KC set the
song actually activated. The profile is seed-stable to r > 0.97.

Note on raw (not lesion-referenced) correlations: r(dS_learn, dS_lesion) is -0.70 to -0.77
because the shared seed noise at the touched DNs happened to be positive; Spearman is -0.06.
Always report E, never raw dS, and say why.

## What this does and does not license
May say: "same result in 5 of 5 seeds"; "cutting the dopamine rule removes it (0.1 +- 0.3 Hz)";
"random synapse pairing removes ~70% of it"; "reversing reward and punishment reverses the
output of the descending neurons (r = -0.91)"; "52 descending-neuron types changed".
Must not say: "behaviour" (still open-loop rate readout, US given regardless of what the
network does); "learned to avoid misery" (approach drive fell, avoidance did not rise; unchanged);
"song-specific" (Mozart shift +5.66 is larger than skyhigh's).
Not done: reversed arm at seeds 1-4 (one run each if a reviewer asks); Tully-Quinn reversal
within one brain; trained-readout-on-frozen-brain control; eta 1e-5 ramp probe.
