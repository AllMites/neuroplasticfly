"""Run a command once one of N GPU slots is free. Parallel agents share one GPU this way.

Usage: .venv/Scripts/python.exe tools/gpu_slot.py -- <cmd> [args...]
Env GPU_SLOTS (default 2: two concurrent brain sims were the tested ceiling on one 16 GB GPU).
Lock files + 5 s polling, no queue fairness. Liveness of a stale lock's owner uses the Win32 API on Windows
and os.kill(pid, 0) elsewhere. Optional: every command it wraps also runs without it.
"""
import ctypes
import shutil
import os
import subprocess
import sys
import time

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gpu_slots")


def alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def acquire(n):
    os.makedirs(DIR, exist_ok=True)
    while True:
        for i in range(n):
            p = os.path.join(DIR, f"slot{i}.lock")
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return p
            except FileExistsError:
                try:
                    if not alive(int(open(p).read() or 0)):
                        os.remove(p)  # stale: owner died without cleanup
                except (OSError, ValueError):
                    pass
        time.sleep(5)


def main():
    cmd = sys.argv[sys.argv.index("--") + 1:]
    cmd[0] = shutil.which(cmd[0]) or os.path.abspath(cmd[0])  # CreateProcess ignores relative paths
    lock = acquire(int(os.environ.get("GPU_SLOTS", "2")))
    print(f"[gpu_slot] {os.path.basename(lock)} acquired", flush=True)
    try:
        sys.exit(subprocess.call(cmd))
    finally:
        os.remove(lock)


if __name__ == "__main__":
    assert alive(os.getpid()) and not alive(999999)  # self-check of the liveness probe
    main()
