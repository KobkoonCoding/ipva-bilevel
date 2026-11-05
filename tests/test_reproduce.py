"""The table script and the checks of the numbers that need only the stored results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> str:
    done = subprocess.run([sys.executable, str(ROOT / "reproduce" / script), *args], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_tables_agree_with_the_numbers_of_the_paper():
    run("make_tables.py")
    assert run("check_numbers.py", "--stored-only").splitlines()[-1] == "PASS"
