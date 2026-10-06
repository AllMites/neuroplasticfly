"""regime/locator.py on synthetic rates: the signal dies at a known stage, the label comes from a prereg file."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regime import locator as L

rng = np.random.default_rng(0)
x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])          # signed left-minus-right drive, Hz
G, S = x.size, 8
# 4 stages x 20 cells (10 left, 10 right). Gain of the left-right difference per stage: the signal
# survives ORN and uPN, then dies at KC (gain 0); MBON is downstream of the loss, also 0.
gain = {"ORN": 1.0, "uPN": 0.6, "KC": 0.0, "MBON": 0.0}
base = {"ORN": 50.0, "uPN": 20.0, "KC": 5.0, "MBON": 10.0}
rates = np.zeros((G, S, 80))
left, right = {}, {}
for i, st in enumerate(gain):
    lo = 20 * i
    left[st], right[st] = np.arange(lo, lo + 10), np.arange(lo + 10, lo + 20)
    rates[:, :, left[st]] = base[st] + gain[st] * x[:, None, None] / 2
    rates[:, :, right[st]] = base[st] - gain[st] * x[:, None, None] / 2
rates += rng.normal(0, 1.0, rates.shape)

stages, hz = {}, {}
for st in gain:
    stages[st], hz[st] = L.readout(rates, left[st], right[st])

PREREG = """# Preregistration: toy locator

## Label rules (fixed now)
```locator
stages: ORN > uPN > KC > MBON   # pathway order
t_min: 2
silent_hz: 0.1
k: 6                            # >= 6 of 8 seeds share the slope's sign
```
"""

with tempfile.TemporaryDirectory() as d:
    def prereg(text, n=[0]):
        n[0] += 1
        p = os.path.join(d, "PREREGISTER_toy%d.md" % n[0])
        open(p, "w").write(text)
        return p

    # 1. the locator finds the stage where the signal dies, with the label rules read from the file
    p = prereg(PREREG)
    out = L.locate(stages, x, p, mean_hz=hz)
    print(L.table(out))
    assert out["label"] == "LOST AT KC" and out["lost_at"] == "KC", out["label"]
    assert out["rules"]["prereg"] == p and out["rules"]["k"] == 6 and len(out["rules"]["prereg_sha256"]) == 64
    assert abs(out["stages"]["ORN"]["slope"] - 1.0) < 0.05, out["stages"]["ORN"]["slope"]
    assert abs(out["stages"]["uPN"]["ratio_to_first"] - 0.6) < 0.05
    assert out["stages"]["ORN"]["snr"] > 10 and abs(out["stages"]["KC"]["snr"]) < 1

    # 2. the label follows the file, not the code: a stricter t_min in the prereg moves the verdict
    strict = L.locate(stages, x, prereg(PREREG.replace("t_min: 2", "t_min: 1e9")), mean_hz=hz)
    assert strict["label"] == "INSTRUMENT FAIL (ORN)", strict["label"]

    # 3. parallel branches: "uPN | KC" survives if either carries it; the loss moves to MBON
    par = L.locate(stages, x, prereg(PREREG.replace("uPN > KC", "uPN | KC")), mean_hz=hz)
    assert par["label"] == "LOST AT MBON", par["label"]

    # 4. a silent stage is labelled as such
    hz_silent = dict(hz, KC=0.0)
    assert L.locate(stages, x, p, mean_hz=hz_silent)["label"] == "LOST AT KC (SILENT)"

    # 5. carried all the way when every stage keeps the signal
    full = {st: L.readout(rates, left["ORN"], right["ORN"])[0] for st in gain}
    assert L.locate(full, x, p)["label"] == "CARRIED TO MBON"

    # 6. a typo'd rule key, a missing block, or a preregistered stage without data are refused
    for bad in (PREREG.replace("t_min", "tmin"), "# no block\n"):
        try:
            L.read_rules(prereg(bad))
            raise AssertionError("accepted a bad prereg")
        except ValueError:
            pass
    try:
        L.locate({k: v for k, v in stages.items() if k != "MBON"}, x, p)
        raise AssertionError("accepted a missing stage")
    except KeyError:
        pass
print("test_locator OK")
