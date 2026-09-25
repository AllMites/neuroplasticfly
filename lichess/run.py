"""Launch lichess-bot with the fly engine.

lichess-bot insists on loading homemade.py from its own directory and on a
config file with the token inlined, so this copies both into place at start-up
and never writes the token to anything that is not already gitignored.

The BOT upgrade is deliberately NOT done here. Upgrading an account is
irreversible and must be a human decision: see README.md.

Run: docker compose up lichess   (needs LICHESS_TOKEN in .env)
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = "/lichess-bot"
RUNTIME_CONFIG = "/tmp/lichess_config.yml"


def main():
    token = os.environ.get("LICHESS_TOKEN", "").strip()
    if not token:
        sys.exit("LICHESS_TOKEN is empty. Put it in .env (gitignored) as "
                 "LICHESS_TOKEN=lip_xxx with the bot:play scope.")

    ckpt = os.environ.get("FLYCHESS_CKPT", "/app/checkpoints/full/best.pt")
    if not os.path.exists(ckpt):
        sys.exit("no checkpoint at %s; train one before going online." % ckpt)

    shutil.copy(os.path.join(HERE, "homemade.py"), os.path.join(BOT_DIR, "homemade.py"))
    config = open(os.path.join(HERE, "config.yml")).read()
    config = config.replace("${LICHESS_TOKEN}", token)
    with open(RUNTIME_CONFIG, "w") as fh:
        fh.write(config)

    os.makedirs(os.environ.get("FLYCHESS_RECORD", "/app/games/lichess"), exist_ok=True)
    print("starting lichess-bot with", ckpt, flush=True)
    raise SystemExit(subprocess.call(
        [sys.executable, "-u", "lichess-bot.py", "--config", RUNTIME_CONFIG, "-v"],
        cwd=BOT_DIR))


if __name__ == "__main__":
    main()
