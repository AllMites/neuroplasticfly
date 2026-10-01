# Chunk-2 independent evaluation (Opus, 2026-10-01) + log hashes

All 18 per-base labels recomputed independently from the jsonl: match. rung A PASS CONFIRMED; rung 3 PASS CONFIRMED; arm ii 5/5, 5/5, 0/5 CONFIRMED.
Bidir FAIL = LABEL-CORRECT-BUT-MISLEADING: relearn 3/3 scorable cycles on every base (timed > depress every cycle); fails only the F_1 floor (0.131 vs 0.486) borrowed from rung A. Dose: one F block ~0.14-0.28x rung A's 30-trial dose (code); measured |w_mean_frac| at F1 = 0.30x (c0 timed), 0.45x (c1), 0.20x (depress); dose-matched rung A drop ~0.16 vs bidir 0.131 (~0.8x). Floor needs ~2.6x the F1 dose. NEEDS-SPIKING is the diagnostic fallback, NOT evidence for spiking.

Paper wording: Under the open-loop MB schedule the timed rule re-learned in 3 of 3 scorable cycles on every base. The preregistered F_1 floor, borrowed from rung A's 30-trial dose, failed (0.13 vs 0.49). One F block delivers about a third of rung A's weight change, and at matched weight change the rate model's drop is within 20% of rung A's. We therefore report bidir as untested against a dose-matched reference, not as needing spiking.

Ceiling: attractor-held approach MBONs are plastic (53% drop vs 59% others); hard ceiling 0.677 > threshold 0.486; not limiting. Rate 0.575 vs LIF 0.971 = linear readout vs thresholded LIF; still rising at trial 30.
Arm ii vs c0: rung A/3 identical to 3rd decimal; regression claim holds for rung A and rung 3. Bidir differs: tonic PAM ~1.4 vs 0.31 Hz, per-tick timed rule -> c1 avoid MBONs collapse 5.27 -> ~0.22 Hz by B4 (c0 4.24); |w_mean_frac| 1.5x. Do NOT extend 'gains do not break paper 0' to per-tick rules.
Provenance clean: 24 runs at 2fe2f2d 07:40-07:59 after prereg commit 07:39, clean trees, no overwrites, no aborts, 0 guard breaches.

Follow-up (new prereg): identical open-loop bidir schedule on paper-0 LIF (eln8, v783, 5 seeds), timed + depress -> real MB-level F_1 reference; re-score rate at 0.5x of it; dose-matched rung A comparison as secondary, fixed before launch; report c1 avoid-side collapse.

## Log sha256 (logs gitignored; archived by hash)

- bidir_depress_c0.jsonl c4859d7467fe31291d7b66953c785ddea76055bd6c6929ee827ff789c7e585dd
- bidir_depress_c1s0.jsonl 8543e71f8bbb95d73fb181032f1d08f2d3aaf73542eecf65c525abceaa377ce0
- bidir_depress_c1s1.jsonl c4f2160e03b82b7a78029c38108a8764031dc3beb7b753af056c5345363ee9a3
- bidir_depress_c1s2.jsonl d3069b49de607fb524fb64cba4b4ed62847e74c19e7c0d7ec64c5ac4a6a9ca8a
- bidir_depress_c1s3.jsonl cc4d22cd28aae6bf4c569d7e367b5b9942957403f585980cca50e77bb12e6fb0
- bidir_depress_c1s4.jsonl 83d8e2940eb2937eb33d79da404b36bce15b74742c24ecae61a14553bee32352
- bidir_timed_c0.jsonl ede25009dd41a2742463f3abb81bd6db913cb3d40ada2c3d6f6372b5dd0286e0
- bidir_timed_c1s0.jsonl fa6cb3044df4eec92b83ecc4c51a3c716fca36ae058276143b9d84fbc47e07e8
- bidir_timed_c1s1.jsonl 73f59b1f1c7b23e629ed8ee0a179811b544bd175dd82740bb093f8b235d5aed3
- bidir_timed_c1s2.jsonl 1ec8a010fba9dc5dc86a561e45a3b52e369c9e8c23f8cae374c8c588737e4f6e
- bidir_timed_c1s3.jsonl 644b3f203fa90407904b510b8d13aed13658e02ab41d7fe8297bd9afce311aeb
- bidir_timed_c1s4.jsonl 33ada050cb607e6f66ae4bc741527a4d75bae099f7ce367dffac87a4795c1a88
- gate_c0.jsonl 4efd46ea8365cf93cfbcba25139033f13700b1ecf84b234fdd9dfe5c04c76062
- gate_c1s0.jsonl 32a48819af0c3333ec965afe3514b8858f6298b917193ed30e903a939765c6b2
- gate_c1s1.jsonl 3e23ff8697569dce247730c5e5e5de7a389a4a372c35ca9789c2a476c8d5e11f
- gate_c1s2.jsonl f5f9799cf35e7a23988035857cc06b198f808c8ec9502d4ce24f7382b6449321
- gate_c1s3.jsonl 06964465f3ae16a8ad96bfccf9c06a010c989b400a9fd8d9c08b76e0c292edcb
- gate_c1s4.jsonl dcb876edc8256b33dbdbe01de923641e2befbac2fdc72ad22d195953cb138ba5
- rung3_c0.jsonl 1395564df5a9343aedd9ab81e9c08390154a01d8027f55d1f8c14aeaf359e6ff
- rung3_c1s0.jsonl 3dc3d0e260354c6766f79361568b50d2f6de748a6edf0f6a38a27b17bbd66c6f
- rung3_c1s1.jsonl 935828f2002dc569eab609d563bf81de38ce9739a797a6330bf4181288f73659
- rung3_c1s2.jsonl 13d6c50e8f6fca20c99597921d60ce541e0206dffac7875ff8b655b399fa85f9
- rung3_c1s3.jsonl 066157d3d2a02da5332cb89ffb2436edb68eb0d6bcd61c5ac22dddec4b3f3628
- rung3_c1s4.jsonl 010d02311f177a77f8931495fafa79847c1cc9d690dce658dc16d7aeb44182fe
- rung_a_c0.jsonl 3382cfc7df23955bd45dc2ba15f42ae610d5484afbd5700a0cc3e8fe812e6839
- rung_a_c1s0.jsonl 41746977ea7c5ece2657e4ea275ca426a95f5522d749987a9a06069b3ef99142
- rung_a_c1s1.jsonl 7cf6839859cf664d5224772d82ca2f78c97fd45c3b1cf8d16e4aeabf4af570a8
- rung_a_c1s2.jsonl 6ffef590bb1d80cd14220bd51c57834c5ee755b3a9eabba45bacb6479998e2b5
- rung_a_c1s3.jsonl e9810d7ba618a71dcee1c13b42a5e70561cc0213c46f637b6f639c7a2400888c
- rung_a_c1s4.jsonl d4be72bbdaacac939fd5b959991f04e5d79529e935730c2e4011796d13b51c16
