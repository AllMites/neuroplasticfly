# PREREGISTER: chunk-1 fit PASS 2 (POST-HOC) + ignition loop test

Written and committed 2026-09-24 ~23:10 Manila, BEFORE the primary chunk-1 batch finished and before any
pass-2 or loop-test run. User-approved ("add that second pass with a loop test"). Primary prereg:
PREREGISTER_rate_chunk1_fit.md (2838bfe + amendments 1-2); its label is unaffected by this file.

**POST-HOC status.** Pass 2 was designed after seeing ONE primary cell (cap 2, seed 0; H37): the fit
term took only the values 9.0 and 4.5, because two reward compartments ignited together and then
fell back. Every pass-2 result is labelled POST-HOC in any write-up. The primary label stays the
preregistered answer.

## Part 1: pass 2 (optimiser suited to all-or-none steps)

- Identical to the primary run (engine, owned targets, 933 edge-type gains in [0, cap], surrogate on
  owned targets with beta 35.0282, loss margins, Adam lr 0.1, 400 iterations, caps
  2..256 x seeds 0-4, battery, g*, labels, bound 18.287, ensemble qualifier), EXCEPT:
  1. **Pull weight 0** (was 0.01). This removes the diagnosed mechanism: once a pair ignites, its
     hinge term has zero gradient, and the pull toward gain 1 dragged it back down.
  2. **Keep-best.** The gains of the iteration with the lowest fit term are restored before the
     battery runs (ties: the earliest iteration). Early stopping when F1-F5 are satisfied is
     unchanged.
- Cells also store raw edge_x (the loop test needs the exact gains).
- Output: results/rate_chunk1_fit_pass2/. Labels as in the primary prereg, prefixed "PASS2 (POST-HOC)".
- Known cost: without the pull, gains are free to drift up to the cap. g* and the top-gains table
  report this. The minimal-change preference is lost, and the write-up must say so.

## Part 2: loop test (no fitting)

**Fits tested.** For each seed 0-4: the pass-2 cell at the LOWEST cap where at least one R compartment
has D >= 1.0 Hz under FOX after fitting. If a seed has no such cell, it is skipped. If no seed has one:
label NO-IGNITION, and Part 2 ends.

**(a) Step vs graded, with hysteresis.** Uncut fit.
- Fox drive swept 0, 10, ..., 200 Hz up, then 200 -> 0 down. 300 ms per step, state carried between
  steps (engine carry is exact).
- Readout: D_R (mean over R neurons) of each step's window.
- **BISTABLE:** max over drive levels of |D_R(up) - D_R(down)| >= 1.0 Hz.
- Else **STEP:** the largest increase between adjacent levels on the up-sweep is >= 50 % of
  (max D_R - min D_R) on the up-sweep, and that range is >= 1.0 Hz.
- Else **GRADED.** If the range is < 1.0 Hz: **FLAT**.
- Per fit; the summary is the majority label over the tested fits.

**(b) Which connections sustain it.** FOX 150 Hz, 1000 ms, one cut at a time. Readout: n_up = number of
R compartments with D >= 1.0, and D_R.
- C1: PAM -> PAM edge-type gains set to 0.
- C2: PAM -> FDA (feedback) gains set to 0.
- C3: FDA -> FDA gains set to 0.
- C4: C1 + C2 + C3 (every recurrent edge within the owned pools).
- C5: output-silence every non-owned neuron active (rate > 0) under FOX in that fit, except the
  driven Fox. This cuts every loop through the rest of the brain, but ALSO every non-owned
  feedforward relay, so it is read as "needs the rest of the brain", not as "loop".
- A cut KILLS ignition in a fit if n_up drops to 0.
- **Mechanism label** (majority over the tested fits):
  - OWNED-LOOP: C4 kills; the loop is named by whichever of C1-C3 kills alone, or "distributed" if
    none does.
  - NEEDS-REST-OF-BRAIN: C4 does not kill, C5 does.
  - FEEDFORWARD: neither C4 nor C5 kills. Ignition is a steep feedforward threshold, not a loop.

## Artefact caveat (stated before the result)

The rate model has no noise, is linear above threshold, and has a hard 454.5 Hz ceiling. BISTABLE or
STEP here is a property of this model, which can over-state all-or-none switching compared with
noisy, graded neurons. Testing with noise is NOT in scope; if the result matters for a paper, it
needs its own prereg.

## Run order

Pass 2 runs on GPU slot 1 at the same time as the primary batch on slot 0. The loop test runs
automatically when pass 2 ends. Each is analysed once.
