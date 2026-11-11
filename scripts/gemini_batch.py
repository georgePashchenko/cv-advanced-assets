"""Run gemini_edit.py for many jobs in parallel: gemini_batch.py <jobs.txt>.

Each line of jobs.txt: <image or -> <prompt.txt> <out> [aspect]. Existing outputs are skipped.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [ln.split() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]


def run(job):
    if os.path.exists(job[2]):
        return job[2] + " (exists)"
    r = subprocess.run([sys.executable, os.path.join(HERE, "gemini_edit.py"), *job], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()[-300:]


with ThreadPoolExecutor(8) as ex:
    for out in ex.map(run, jobs):
        print(out)
