# RESULT: rate model chunk-1 unfitted control (path 2, PRD phase 3)

Prereg `194be67`; engine `d372b1b`; script `975464e`. Single deterministic run per condition. `result.json` here.

## Labels
- **Arm A (primary, LIF-matched): FAILS-UNFITTED.** 3/11 rows pass (F5, H3, H5): the three "no response" rows. They pass trivially, because nothing responds.
- **Arm B (secondary, Lappalainen default): FAILS-UNFITTED.** Same 3/11.
- Gates: G0 Fox -> CB0233 = **217 synapses over 4 edges** at floor 5, which matches Christie Data S2A (217) exactly and supports Fox = CB0525. G1 and G2 pass. S passes in both arms (0 % of neurons at r_max; max 452.7 Hz in A, 150 Hz, the drive clamp, in B).
- **Decision (prereg rule): the chunk-1 fit goes ahead** (PRD phases 4-5), starting from these default parameters.

## What happens (descriptive)
- Arm A, Fox drive at 150 Hz: FDA-I 1.2 Hz, FDA-II 6.2 Hz. Every reward-set (R) PAM stays below threshold: D = 0.000 exactly.
- Arm A, sugar drive: Fox reaches 9.2 Hz, FDA-I 0.0, FDA-II 0.14. R receives no positive input at all.
- Arm B: the path exists but attenuates at every hop. R mean rate is 9.7e-5 Hz under Fox and 1.5e-6 Hz under sugar, about 10^4-10^6 below the 1 Hz "up" threshold.

## Post-hoc diagnostic (computed after the label; changes no label)
Synaptic drive onto the 216 R neurons in arm A, Fox condition:
- 28 / 216 R neurons get any net positive input.
- Best neuron: 2.06 Hz of input against a 35.03 Hz threshold, so a **gain of about 17x** on that input would be needed just to reach threshold on the single best neuron.
- Sugar condition: 0 / 216 R neurons get positive input, because FDA-I is silent under sugar.

## Reading
- The rate reduction does NOT route sugar to reward PAMs. This agrees with the LIF result (the path dies at FDA-I -> PAM, 0.81 % of PAM input) and with the compartment gate (D42). It is now negative across three neuron models: point LIF, passive compartment and graded rate.
- Arm B removes the threshold completely and still gets 10^-4 Hz. The block is attenuation along a sparse, multi-hop path, not only a spiking threshold.
- Consequence for chunk 1: a fit would need a large type-level gain (on the order of 17x or more at the FDA -> PAM hop, plus enough to lift FDA-I under sugar). The chunk-1 fit prereg must report every fitted gain next to this baseline, and must say in advance which gain sizes it would call implausible.

## Caveats
Floor 5 only. A single drive rate (150 Hz). The arm-A linearisation has 20.6 Hz rms error near threshold, and there is no noise, so there is no firing below threshold. The noisy LIF can fire below threshold; this reduction cannot.
