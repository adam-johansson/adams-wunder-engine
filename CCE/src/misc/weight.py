import math
import subprocess
import re
from pathlib import Path
import math

WEICO_DIR = Path.cwd() / "weico"          # the weico subfolder next to where you run the script
WEICO_EXE = WEICO_DIR / "weico.exe"
INPUT_FILE = WEICO_DIR / "PerformanceResults.txt"
OUTPUT_FILE = WEICO_DIR / "weightOutput.txt"
TEMPLATE_FILE = WEICO_DIR / "PerformanceResults_template.txt"   # original file, never modified


TO_POINT = "Take-off"
TO_BYPASS_FLOWS = ["W18"]          # add "W13", "W14" to change the whole bypass consistently


POINT = "Top-of-climb"       # only this point is changed; Take-off is kept from the file

# weico station -> index in your arrays. CHECK these, especially the turbine stations.
STATION_MAP = {
    "2": 2,      # 2   before fan
    "23": 6,     # 21  (core inlet behind fan)
    "24": 6,     # 21  (LPC inlet; same as 23 in the template)
    "25": 7,     # 25  LPC outlet
    "26": 8,     # 26  HPC inlet  after intercooler
    "3": 9,      # 3   HPC outlet
    #"4": 14,     # 4   HPT INLET (secondary burner outlet = turbine inlet) just use the reference engine data here
    #"43": 14,    # 5   HPT oulet    (after rotor cooling)   will just use LPT inlet conditions here
    "46": 14,    # 5   LPT inlet                        
    "5": 17,     # 5   LPT outlet                        
    "8": 18,     # 8   Nozzle inlet
    "13": 3,     # 13  bypass duct inlet
    "14": 3,     # 13  somewhere in the bypass   #I dont have this station in the CCE
    "18": 4,     # 14  (bypass nozzle)         #this is after pressure loss in the bypass
}

# weico power entry -> key in power_dict. PW43 (HPT) is kept from the template.
# NOTE: Adding power did not change the weight
POWER_MAP = {
    "2": "fan",     # PW2  fan power
    "24": "lpc",    # PW24 LPC power
    "26": "hpc",    # PW26 HPC power
    "5": "lpt",     # PW5  LPT power
}

# NOTE: Did also not change the weight
# weico area entry -> key in area_dict [m2]
AREA_MAP = {
    "8": "A8",               # A8  core nozzle
    "18": "A_bypass_tot",    # A18 bypass nozzle = A18 + A17 + A_IC
}



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

    return

def fortran_e(x):
    """Format a number like the cycle code does: 0.25672984967476E+03."""
    if x == 0:
        return "0.0"
    exp = math.floor(math.log10(abs(x))) + 1
    if round(abs(x) / 10**exp, 14) >= 1.0:
        exp += 1
    return f"{'-' if x < 0 else ''}{abs(x) / 10**exp:.14f}E{exp:+03d}"

def make_weico_input(p_array, T_array, f_array, m_array, power_dict, area_dict, bpr):
    """Write p [Pa], T [K], FAR, mass flow [kg/s], powers [W] and areas [m2] for POINT
    into PerformanceResults.txt. At TO_POINT, the bypass flows are set from the
    template W2 and bpr. Unmapped entries keep their template values."""
    with open(TEMPLATE_FILE, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)

    arrays = {"P": p_array, "T": T_array, "FAR": f_array, "W": m_array}
    point = None
    w2_to = None
    for i, line in enumerate(lines):
        if line.startswith("Point name:"):
            point = line.split(":", 1)[1].strip()
            continue
        m = re.match(r"^(PW|P|T|FAR|W|A)(\d+):(\s*)(\S+)(\s*)$", line)
        if not m:
            continue
        kind, station = m.group(1), m.group(2)

        if point == TO_POINT:
            if kind + station == "W2" and w2_to is None:
                w2_to = float(m.group(4))                 # take-off W2 from the template
            elif kind + station in TO_BYPASS_FLOWS:
                value = w2_to * bpr / (1 + bpr)
                lines[i] = f"{kind}{station}:{m.group(3)}{fortran_e(value)}{m.group(5)}"
                print(f"Take-off {kind}{station}: {float(m.group(4)):.2f} -> {value:.2f} kg/s")
            continue

        if point != POINT:
            continue
        if kind == "PW" and station in POWER_MAP:
            value = power_dict[POWER_MAP[station]]
        elif kind == "A" and station in AREA_MAP:
            value = area_dict[AREA_MAP[station]]
        elif kind in arrays and station in STATION_MAP:
            value = arrays[kind][STATION_MAP[station]]
        else:
            continue                       # not mapped -> keep template value
        lines[i] = f"{kind}{station}:{m.group(3)}{fortran_e(value)}{m.group(5)}"

    with open(INPUT_FILE, "w", newline="") as fh:
        fh.write("".join(lines))


# Section headers in weightOutput.txt, used to keep repeated labels apart
SECTIONS = [
    "Fan output", "Imc output", "Ipc output", "imc2 output", "Hpc output", "Comb output",
    "HPT output", "Tcf output", "LPT output", "Tec output", "hot nozzle output",
    "Bypass duct output", "cold nozzle output", "Low pressure shaft", "High pressure shaft",
    "Nacelle output", "Bearings output", "Engine summary output", "Estimated component efficiencies",
]
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?|NaN"


def _split_sections(text):
    """{section name: text from its header up to the next header}."""
    starts = []
    for name in SECTIONS:
        m = re.search(r"^" + re.escape(name), text, re.MULTILINE)
        if m:
            starts.append((m.start(), name))
    starts.sort()
    ends = [s for s, _ in starts[1:]] + [len(text)]
    return {name: text[s:e] for (s, name), e in zip(starts, ends)}


def _nums(section, label, last=False):
    """Numbers on the first (or last) line in section that starts with label."""
    lines = re.findall(r"^\s*" + re.escape(label) + r"\s*:?(.*)$", section, re.MULTILINE)
    if not lines:
        raise ValueError(f"'{label}' not found in weightOutput.txt")
    return [float(x.replace("D", "E").replace("d", "e")) for x in re.findall(NUMBER, lines[-1 if last else 0])]


def _annulus(r_out, r_in):
    return math.pi * (r_out**2 - r_in**2)


def read_weico_output():
    """Read weightOutput.txt and return weights [kg], lengths [m], radii [m] and areas [m2]."""
    text = OUTPUT_FILE.read_text()
    sec = _split_sections(text)
    summ = sec["Engine summary output"]

    total = _nums(summ, "TotalEngineWeightIs")[0]
    if total != total:            # NaN check
        raise ValueError("weico returned NaN for the total engine weight")

    # module weights and stage numbers between "Dry weight is" and "Total engine length"
    modules, stages = {}, {}
    block = summ.split("Dry weight is", 1)[1].split("Total engine length", 1)[0]
    for line in block.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], float(parts[1])
        if name.endswith("StageNo"):
            stages[name[:-len("StageNo")]] = int(value)
        elif name != "LastStageBladeHeight":
            modules[name] = value

    fan, imc, ipc, hpc = sec["Fan output"], sec["Imc output"], sec["Ipc output"], sec["Hpc output"]
    hpt, tcf, lpt, nac = sec["HPT output"], sec["Tcf output"], sec["LPT output"], sec["Nacelle output"]

    ipc_rt, ipc_rh = _nums(ipc, "tip radius", last=True), _nums(ipc, "hub radius", last=True)   # stage boundaries
    hpc_rt, hpc_rh = _nums(hpc, "tip radius", last=True), _nums(hpc, "hub radius", last=True)

    radii = {   # (outer, inner) per station [m]
        "2":  (_nums(fan, "tipRadius")[0], _nums(fan, "hub radius")[0]),                       # fan face
        "13": (_nums(fan, "rotor tip exit radius")[0], _nums(fan, "splitter_radius")[0]),     # bypass entry
        "21": (_nums(fan, "splitter_radius")[0], _nums(fan, "rotor hub exit radius")[0]),     # core entry behind fan
        "24": (_nums(imc, "main stream strut exit rt")[0], _nums(imc, "main stream strut exit rh")[0]),  # IPC inlet
        "25": (ipc_rt[-1], ipc_rh[-1]),                                                       # IPC exit
        "26": (hpc_rt[0], hpc_rh[0]),                                                         # HPC inlet
        "3":  (hpc_rt[-1], hpc_rh[-1]),                                                       # HPC exit
        "4":  (_nums(hpt, "tip radius")[0], _nums(hpt, "hub radius")[0]),                     # HPT stage 1
        "46": (_nums(tcf, "exit tip radius")[0], _nums(tcf, "exit hub radius")[0]),           # LPT inlet
        "5":  (_nums(lpt, "Tip radii")[0], _nums(lpt, "Hub radii")[0]),                       # LPT exit
    }
    areas = {"A" + st: _annulus(ro, ri) for st, (ro, ri) in radii.items()}
    d_hilite = _nums(nac, "D_hilite")[0]
    areas["A_hilite"] = math.pi * d_hilite**2 / 4                                             # nacelle inlet capture area

    areas["A_nacelle_max"] = math.pi * _nums(nac, "NacelleDiameter")[0]**2 / 4               # nacelle frontal area
    areas["A_nacelle_exit"] = math.pi * _nums(nac, "De")[0]**2 / 4                           # nacelle exit (De)

    comb = sec["Comb output"]
    d = {k: _nums(comb, k + " diameter")[0] for k in ("outer casing", "outer liner", "inner liner", "inner casing")}
    areas["A_comb_casing"] = math.pi / 4 * (d["outer casing"]**2 - d["inner casing"]**2)    # combustor casing annulus
    areas["A_comb_liner"] = math.pi / 4 * (d["outer liner"]**2 - d["inner liner"]**2)       # combustor liner annulus

    stage_areas = {   # annulus areas at every stage boundary, inlet first [m2]
        "ipc": [_annulus(ro, ri) for ro, ri in zip(ipc_rt, ipc_rh)],
        "hpc": [_annulus(ro, ri) for ro, ri in zip(hpc_rt, hpc_rh)],
        "lpt": _nums(lpt, "A:"),
    }

    return {
        "total_weight": total,                                   # kg
        "dry_weight": _nums(summ, "Dry weight is")[0],           # kg
        "module_weights": modules,                               # kg
        "stages": stages,
        "total_length": _nums(summ, "Total engine length is")[0],    # m
        "nacelle_diameter": _nums(nac, "NacelleDiameter")[0],        # m
        "nacelle_length": _nums(nac, "NacelleLength")[0],            # m
        "fan_diffusion_factor": _nums(fan, "fan diffusion factor")[0],
        "fan_Utip": _nums(fan, "Utip")[0],                           # m/s
        "fan_rpm": _nums(fan, "xnl (second data is rpm)")[1],
        "radii": radii,                                          # m, (outer, inner)
        "areas": areas,                                          # m2
        "stage_areas": stage_areas,                              # m2, lists per component
    }


def calculate_powerplant_weight(p_array, T_array, f_array, m_array, power_dict, area_dict, mach_dict):

    print(power_dict)
    print(area_dict)

    p_array = p_array * 1000

    m2 = m_array[2]
    m21 = m_array[6]

    bpr = m2/m21 - 1

    make_weico_input(p_array, T_array, f_array, m_array, power_dict, area_dict, bpr)
    run_weico()
    info = read_weico_output()

    area_dict = {**area_dict, **info["areas"]}     # my nozzle areas + weico geometry
    w_no_hpt = info["total_weight"] - info["module_weights"]["Hpt"]
    A_fan, A13 = info["areas"]["A2"], info["areas"]["A13"]

    print(f"Total engine weight: {info["total_weight"]:.2f} kg")
    print(f"Weight no HPT: {w_no_hpt:.2f} kg")
    print(f"Fan area: {A_fan:.2f} m2")
    print(f"A13: {A13:.2f} m2")
    print(f"A18: {area_dict['A18']:.2f} m2")
    print(f"A8: {area_dict['A8']:.2f} m2")
    print(f"A17 (oil cooler): {area_dict['A17']:.2f} m2")
    print(f"A IC cooler: {area_dict['A_IC']:.2f} m2")

    print(area_dict)


    return w_no_hpt, area_dict

    