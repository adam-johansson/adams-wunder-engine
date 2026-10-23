import math
import re
import numpy as np
import matplotlib.pyplot as plt

from weico_runner import run_weico, INPUT_FILE

POINT = "Take-off"
#POINT = "Top-of-climb"
LINE_RE = re.compile(r"^(\s*)(\w+):(\s*)(\S+)(\s*)$")


# ---------- reading / editing PerformanceResults.txt ----------

def read_file(path):
    with open(path, newline="") as f:          # keep original line endings
        return f.read()


def write_file(path, text):
    with open(path, "w", newline="") as f:
        f.write(text)


def fortran_e(x):
    """Format like the cycle code does: 0.25672984967476E+03."""
    if x == 0:
        return "0.0"
    exp = math.floor(math.log10(abs(x))) + 1
    mant = abs(x) / 10**exp
    if round(mant, 14) >= 1.0:                 # rounding pushed 0.99999... up to 1.0
        exp += 1
        mant = abs(x) / 10**exp
    sign = "-" if x < 0 else ""
    return f"{sign}{mant:.14f}E{exp:+03d}"


def _lines_in_point(text, point):
    """Yield (index, line) for the lines belonging to the named operating point."""
    current = None
    for i, line in enumerate(text.splitlines(keepends=True)):
        if line.startswith("Point name:"):
            current = line.split(":", 1)[1].strip()
        elif current == point:
            yield i, line


def get_values(text, point):
    values = {}
    for _, line in _lines_in_point(text, point):
        m = LINE_RE.match(line.rstrip("\r\n"))
        if m and m.group(2) not in values:
            try:
                values[m.group(2)] = float(m.group(4).replace("D", "E"))
            except ValueError:
                pass                           # text entries like "Configuration: GTFSR"
    if not values:
        raise ValueError(f"Point '{point}' not found")
    return values


def set_values(text, point, changes):
    """Return text with {key: new_value} replaced in one point only.
    Keys that appear twice in the block (e.g. W2) are replaced everywhere."""
    lines = text.splitlines(keepends=True)
    found = set()
    for i, line in _lines_in_point(text, point):
        body = line.rstrip("\r\n")
        eol = line[len(body):]
        m = LINE_RE.match(body)
        if m and m.group(2) in changes:
            key = m.group(2)
            lines[i] = f"{m.group(1)}{key}:{m.group(3)}{fortran_e(changes[key])}{m.group(5)}{eol}"
            found.add(key)
    missing = set(changes) - found
    if missing:
        raise KeyError(f"Not found in point '{point}': {sorted(missing)}")
    return "".join(lines)


# ---------- sensitivity sweep ----------

def sweep(keys, factors=(0.50, 0.75, 1.0, 1.25, 1.50)):
    """Scale each key (one at a time) by each factor and record engine weight."""
    original = read_file(INPUT_FILE)
    base = get_values(original, POINT)
    results = {}
    try:
        w0 = run_weico()
        print(f"Baseline weight: {w0:.2f} kg")
        for key in keys:
            weights = []
            for f in factors:
                if f == 1.0:
                    weights.append(w0)
                    continue
                write_file(INPUT_FILE, set_values(original, POINT, {key: base[key] * f}))
                try:
                    weights.append(run_weico())
                except Exception as e:
                    print(f"  {key} x {f}: failed ({e})")
                    weights.append(np.nan)
            results[key] = np.array(weights)
            print(f"{key:>5}: " + "  ".join(f"{w:9.2f}" for w in weights))
    finally:
        write_file(INPUT_FILE, original)       # always restore the original file
    return w0, np.array(factors), results


def report(w0, factors, results):
    # Sensitivity = (% change in weight) / (% change in input), from the outermost points
    i_lo, i_hi = 0, len(factors) - 1
    dx = factors[i_hi] - factors[i_lo]
    sens = {k: (w[i_hi] - w[i_lo]) / w0 / dx for k, w in results.items()}

    print("\nSensitivity (dW/W) / (dx/x), largest first:")
    for k, s in sorted(sens.items(), key=lambda kv: -abs(np.nan_to_num(kv[1]))):
        print(f"  {k:>5}: {s:+.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, w in results.items():
        if np.nanmax(np.abs(w - w0)) > 1e-6 * w0:   # skip variables with no effect
            ax.plot((factors - 1) * 100, (w / w0 - 1) * 100, "o-", label=k)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"Change in {POINT} variable [%]")
    ax.set_ylabel("Change in total engine weight [%]")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    text = read_file(INPUT_FILE)
    vals = get_values(text, POINT)

    # All station pressures, temperatures and mass flows at take-off (P3, T4, W2, ...).
    # Excludes PW* (powers) and FAR*. Edit this list to study fewer variables.
    #keys = [k for k in vals if re.fullmatch(r"[PTW]\d+", k) and vals[k] != 0]
    #keys = ["P2", "P23", "P25", "P3", "P43", "P3", "P3"]
    #keys = [k for k in vals if re.fullmatch(r"P\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"T\d+", k)]
    keys = [k for k in vals if re.fullmatch(r"W\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"PW\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"A\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"FAR\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"M\d+", k)]
    #keys = [k for k in vals if re.fullmatch(r"A\d+", k)]
    #keys = ["NL", "NH"]
    w0, factors, results = sweep(keys)
    report(w0, factors, results)