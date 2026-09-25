# flypet's image (flypoke + FlyWire v783 sim) plus the chess toolchain.
# Torch is CPU-only on purpose: the sim is the bottleneck and the head is tiny,
# so everything except host-side imitation training runs on CPU in here.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git stockfish \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "git+https://github.com/vshapenko/flypoke" \
    "matplotlib>=3.8" \
    "chess==1.11.2" "zstandard>=0.22" "requests>=2.31" "pyarrow>=16"

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Pinned so a lichess-bot API change cannot break a running bot mid-rating-run.
ARG LICHESS_BOT_SHA=df7e730de58cc3ef2f1415a0dc2eeda842d39167
RUN git clone https://github.com/lichess-bot-devs/lichess-bot /lichess-bot \
    && cd /lichess-bot && git checkout "$LICHESS_BOT_SHA" \
    && pip install --no-cache-dir -r requirements.txt

ENV FLYPOKE_DATA=/data MPLCONFIGDIR=/tmp/mpl PYTHONPATH=/app:/flypet OMP_NUM_THREADS=1
WORKDIR /app
CMD ["python", "-u", "selfcheck.py"]
