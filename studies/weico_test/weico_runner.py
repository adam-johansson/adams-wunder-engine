import subprocess
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]            # two levels up, like your "./../../"
sys.path.append(str(ROOT))              # only needed if you import modules from ROOT

WEICO_DIR = ROOT / "CCE" / "weico"      # adjust to where CCE actually is
WEICO_EXE = WEICO_DIR / "weico.exe"
INPUT_FILE = WEICO_DIR / "PerformanceResults.txt"
OUTPUT_FILE = WEICO_DIR / "weightOutput.txt"


def run_weico(timeout=10):
    """Run weico.exe and return (total engine weight in kg)."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file missing: {INPUT_FILE}")

    # Remove the old output so a failed run can't return stale results
    OUTPUT_FILE.unlink(missing_ok=True)

    result = subprocess.run(
        [str(WEICO_EXE)],
        cwd=WEICO_DIR,            # weico looks for its files in its own folder
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if not OUTPUT_FILE.exists():
        raise RuntimeError(
            f"weico.exe produced no output (exit code {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    text = OUTPUT_FILE.read_text()
    m = re.search(r"TotalEngineWeightIs\s+(\S+)\s+(\S+)", text)
    if not m:
        raise ValueError("TotalEngineWeightIs not found in weightOutput.txt")

    w_kg = float(m.group(1))
    if w_kg != w_kg:              # NaN check
        raise ValueError("weico returned NaN for the total engine weight")
    return w_kg




if __name__ == "__main__":
    kg = run_weico()
    print(f"Total engine weight: {kg:.2f} kg")