# Preregistration: <NAME>

Commit this file before any data exists, then run `python prereg.py freeze` on it (see docs/prereg.md).
Changes after the first run go in a dated **Amendment** section at the bottom, committed before the data they apply to.
Never edit the rules above an amendment.

## Question
<ONE SENTENCE: what the run decides.>

## Prediction
<WHAT YOU EXPECT AND WHY, including the direction. A prediction that turns out wrong is reported as falsified.>

## Design
- Conditions: <e.g. LOOM, RECEDE, DIM>
- Cells / readout: <cell types and side, e.g. LPLC2 left>
- Seeds: <e.g. 1-5>
- Fixed settings: <every constant that matters, e.g. synapse floor, weight scale, frame length. None fitted to the result.>
- Code: <files passed to `freeze --code`>

## Label rules (fixed now)
- PASS: <ordering as pairs, e.g. LOOM > RECEDE and LOOM > DIM, LOOM > 0> on baseline-subtracted rates in >= <k> of <n> seeds
- SUBST: same ordering on raw totals <e.g. spike counts over stimulus frames> in >= <k> of <n> seeds
- MECH (optional): same ordering on <the cell's own input, e.g. summed T4/T5 drive> in >= <k> of <n> seeds
- Labels: PASS / PASS BY RULE, NOT SUBSTANTIVE / PASS, MECHANISM NOT SHOWN / FAIL (from `prereg.verdict`)

## Smoke gate
<A SHORT RUN THAT MUST SUCCEED BEFORE THE FULL RUN, e.g. one seed, check the output has the expected shape and no NaN.
The smoke output is not scored.>

## What each outcome means
- PASS: <claim allowed, worded "in the sim, in our testing">
- PASS BY RULE, NOT SUBSTANTIVE: <reported as an artefact, never as a result>
- FAIL: <where you will look for the break next>

## Amendments
<none yet>
