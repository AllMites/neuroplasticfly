"""lichess-bot homemade engine: every move is one 300 ms fly-brain sim.

Loaded by lichess-bot from inside /lichess-bot, so it imports the flychess
modules off PYTHONPATH (/app). The player is built lazily on the first search so
that importing this module never blocks the bot's startup on building the
139k-neuron network.
"""
import os
import sys

sys.path.insert(0, "/app")

from chess.engine import PlayResult
from lib.engine_wrapper import MinimalEngine

from play import FlyPlayer

CKPT = os.environ.get("FLYCHESS_CKPT", "/app/checkpoints/full/best.pt")
RECORD_DIR = os.environ.get("FLYCHESS_RECORD", "/app/games/lichess")
TEMPERATURE = float(os.environ.get("FLYCHESS_TEMPERATURE", "0.5"))

_player = None


def player():
    global _player
    if _player is None:
        _player = FlyPlayer(CKPT, temperature=TEMPERATURE, record_dir=RECORD_DIR)
    return _player


class FlyEngine(MinimalEngine):
    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        p = player()
        move, top5, res = p.choose(board)
        p.record(board, move, top5, res)
        return PlayResult(move, None)
