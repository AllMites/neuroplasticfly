# Table S3. Learning tests on the bare rate model and on the 5 versions with the reward gains

| Version | Specificity: normalised drop (bar 0.486) | Accumulation: trained / naive (bar 0.429) | Relearning: first-block drop (bar 0.198) | Relearned cycles (of 3) | Avoid MBONs T0 → B4 (Hz) | PAM at T0 (Hz) |
|---|---|---|---|---|---|---|
| bare | 0.575 | 0.968 | 0.131 | 3 | 5.27 → 4.24 | 0.31 |
| gains, start 0 | 0.575 | 0.968 | 0.137 | 3 | 5.27 → 0.23 | 1.40 |
| gains, start 1 | 0.575 | 0.968 | 0.138 | 3 | 5.27 → 0.24 | 1.43 |
| gains, start 2 | 0.575 | 0.968 | 0.137 | 3 | 5.27 → 0.24 | 1.41 |
| gains, start 3 | 0.575 | 0.968 | 0.137 | 3 | 5.27 → 0.21 | 1.30 |
| gains, start 4 | 0.575 | 0.968 | 0.137 | 3 | 5.27 → 0.21 | 1.41 |

Spiking-model references: specificity 0.971, accumulation 0.858 (ratios of the means of 5 noise replicates), first-block drop 0.396 (mean of 5 replicates, SD 0.016, all 5 relearned 3 of 3 cycles).
