# flychess — a fly brain you can watch think

Date: 2026-09-15. Status: approved design, pre-implementation.
Sibling of `../flypet` (FlyWire v783 LIF sim via flypoke, Shiu et al. 2024).
Body movement (flygym / NeuroMechFly) is **spec 2**, out of scope here; this
spec only defines the interface it will consume.

## Goal

Play chess at a verified public rating of roughly 800-1000 Elo (lichess bot
account), with the frozen FlyWire connectome in the decision loop as a
reservoir, and render every neuron's activity per move in 3D. Launch content
angle: "I fed a fly brain sugar every time it played a good chess move".

## Prior art (2026-09-15)

- flychess-hq.vercel.app (Ernesto Lopez, ~09-11): MaleCNS reservoir + linear
  readout, 700 Elo unverified, agadmator/Chess.com coverage. "First
  chess-playing fly" is culturally taken.
- lichess FlyBrainChess (Rioux): MaleCNS-derived NN, rapid 1305.
- cesp99/fly-chess (HF): FlyWire rate net, 37.6% top-1.
- ESA ACT 2025, PMC12109256: Drosophila connectome as reservoir (peer-reviewed
  precedent for our method).

Gaps we fill: real neuron anatomy firing per move (everyone else uses dots or
region heatmaps), verified lichess rating, honest reservoir framing with
ablation, reward circuits shown as real sugar/bitter neurons.

## §1 Architecture (one move)

```
FEN
 -> encode.py: board -> Poisson drives on 4 sensory systems   [authored]
 -> flypoke LIF, full 139,255 neurons, 300 ms, 1 trial        [real wiring, frozen]
 -> rate vector r (139k Hz) -> pooled by cell_type (~3k)       [real]
 -> head: MLP([board planes (+) r_pooled]) -> softmax(1858)    [trained]
 -> mask illegal -> sample
 -> r, spikes, DN table saved -> viz + body interface
```

Sensory encoding (chosen for neuron diversity; amended 2026-09-15 after
measurement, see table below):
- Vision: 8x8 board rasterized onto T4/T5 motion-detector columns (12,246
  neurons, retinotopic), binned by soma position. Own pieces left eye,
  opponent right eye. Piece type -> drive intensity. Photoreceptors R1-6 were
  the original plan and were dropped: their drive dies in the lamina.
- Smell: the high-gain state channel. ~50 ORN glomerulus classes assigned to
  game-state features (piece-type counts, material sign, check, castling,
  side to move). One ORN class alone reaches 27% of the central brain.
- Taste: material balance -> sugar/water (ahead) or bitter (behind) gustatory.
- Touch: in-check or hanging pieces -> Johnston's organ mechanosensors (weak
  propagation, kept for semantics and for the grooming readout).

Measured at 300 ms, 150 Hz Poisson drive, full 139k network (Docker, 1.1 s
per sim):

| channel | neurons | central >1 Hz | descending >1 Hz |
|---|---|---|---|
| R1-6 photoreceptors | 4423 | 0 | 0 |
| T4/T5 | 12246 | 1386 (4.3%) | 14% |
| LC4/LPLC2 | 314 | 798 | 12% |
| ORN_DA1 | 126 | 8911 (27.5%) | 15% |
| sugar / bitter | 129 / 65 | 422 / 119 | 6.6% / 0.8% |
| JO / grooming | 276 / 192 | 38 / 98 | 1-3% |

Sim length 300 ms = 1.1 s wall clock. Do not lengthen the sim; raise drive Hz
or add channels if activity is too sparse.

## §2 Training

Phase A, imitation: Lichess DB, games 1500-2000 Elo, ~200k positions.
Precompute r with 16 parallel sim workers (~4 h at 1.1 s/sim). Train head with
cross-entropy on the human move. Checkpoint per epoch.

Phase B, reinforcement: fine-tune head via REINFORCE, playing Stockfish skill
0-5 and self-play, win/loss reward. Reservoir frozen. After each move the
reward is also *shown*: good move -> sugar gustatory drive (ingestion motor
neurons ~73 Hz, "fed"), bad move -> bitter drive (collapse to 0, "recoils").
Readouts imported from `flypet/behaviors.py`. This is the literal basis of
the launch story.

Improvement curve: every checkpoint plays 100 games vs random + Stockfish
skill ladder; Elo per checkpoint -> `results/elo.json`. Checkpoint 0 is a
genuinely untrained head.

Ablation (mandatory): identical head trained without r. Both Elos reported.
If delta ~ 0, drop the claim "fly wiring helps chess"; keep "fly brain
reacts, trained layer decides".

Compute: RTX 5070 Ti for the head (minutes); 16 cores for sims (bottleneck). Docker Desktop: 32 CPUs, 31 GB.

## §3 Visualization

Per-move artifact `games/<id>/moves/NNN.npz`: `rates` (139k float16),
`spikes` (sparse neuron, t), `fen`, `move`, `top5`, `dn_table` (descending
neuron rates by cell_type and side = body interface for spec 2). ~1 MB.

`dn_table` must always include these readouts (all already measurable in
flypet's stimulus matrix), so spec 2 can drive limbs without re-simulating:
- grooming: DNg29, DNg84 (foreleg rubbing)
- escape / wing: DNp01 giant fiber, DNp04 (jump, wing flick)
- steering / orienting: DNa02, DNa01, DNa03 per side (turn toward board side)
- walking candidates: DNp09, DNa-class (forward drive)
- feeding: ingestion motor neurons (sugar reward), plus bitter collapse
Plus a `sensory_summary` block: per-eye retina drive and gustatory drive, so
the body can orient to the stimulus it was actually given.

Geometry: v783 skeletons, Zenodo 10.5281/zenodo.10877326 (SWC, bulk).
- L0: soma xyz for all 139k from the annotation TSV already in the Docker
  volume (`pos_x/y/z`, `soma_x/y/z`).
- L1: skeletons downsampled to ~50 nodes/neuron, ~7M segments, colored by
  super_class.

Renderer A, web (three.js): point cloud + skeleton lines as GPU buffers,
emission = rate; spike playback 300 ms -> ~3 s. Board beside brain. Hover ->
cell type. Hosts live play against the lichess bot. Deploy Cloudflare
Workers.

Renderer B, Blender: navis -> glTF, emission keyed per frame from `.npz`,
camera orbits for the launch video, HyperFrames for titles/captions.
Blender MCP is available.

Explainer layer: per-move rates grouped by neuropil (39) -> "which regions
light up on a sacrifice" content.

Not now: full meshes, VR, real-time 139k spiking in browser.

## §4 Story

Launch post A: "I fed a fly brain sugar every time it played a good chess
move." Arc: untrained fly plays garbage and gets bitter -> montage -> beats
the author -> "here's what it looks like when it thinks". 60 s on X, ~10 min
YouTube. Follow-ups: B "watch a brain think neuron by neuron", C the honest
ablation post citing the ESA paper. No face required: screen capture,
captions, optional voiceover.

## §5 Testing, honesty, delivery

Runnable checks (one script each, assert-based):
- encode: deterministic; every square hits >= 1 photoreceptor; eyes symmetric.
- sim smoke: start position, 300 ms -> >= 5% of central-brain neurons > 1 Hz.
- reservoir: different positions -> cosine(r1, r2) < 0.99; same seed -> equal.
- head: illegal moves get probability 0; checkpoint 0 vs random ~ 50%.
- ablation runs with every training run.

Elo: lichess BOT account; the lichess rating is the only number quoted.

`WHAT_IS_REAL.md` shipped and linked from every post: real (wiring, rates,
sugar/bitter circuits) / authored (encoding, readout->behavior map) /
trained (head) / absent (the fly learned nothing; the head did).

Repo `flychess/`: `encode.py`, `reservoir.py` (workers), `train.py`,
`play.py` (lichess bot), `viz/` (three.js), `export_blender.py`, reusing the
flypoke Docker image and `flypet/behaviors.py`.

Accepted failure modes: ablation delta ~ 0 (story survives, claim narrows);
reservoir head < 800 Elo (play with board-only head, reservoir for viz,
disclosed).
