import matplotlib.pyplot as plt
import importlib
import numpy as np
from timeit import default_timer as timer
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import os

import sys
sys.path.append("./../../../")

from piston_engine.engine import run_piston_engine
from thermo import fuel_props

# --- Load input file ---
input_file = "4T50ME"
input_dir = "piston_engine.input.EGR_validation"
path = input_dir + "." + input_file
d = importlib.import_module(path)


piston_input = {
    'p_in': d.p_in,
    'T_in': d.T_in,
    'equ_in': d.equ_in,
    'p_ratio': d.p_ratio,
    'cycle': d.cycle,
    'cooling': d.cooling,
    'opposed': d.opposed,
    'cr': d.cr,
    'bore': d.d,
    'bsr': d.bsr,
    'v_mean': d.v_mean,
    'lms': d.lms,
    'Twalls': d.Twalls,
    'ch': d.ch,
    'valve_timings': d.valve_timings,
    'n_valve': d.n_valve,
    'lv_max': d.lv_max,
    'cd': d.cd,
    'eta_c': d.eta_c,
    'mf_tot': d.mf_tot,
    'm_wiebe': d.m_wiebe,
    'phi_sc': d.phi_sc,
    'phi_cd': d.phi_cd,
    'T_fuel': d.T_fuel,
    'p_fuel': d.p_fuel,
    'it': d.it,
    'wiebe_type': d.wiebe_type,
    'valve_type': d.valve_type,
    'far_goal': d.far_goal,
    'cylinders': d.cylinders,
    'fuel': d.fuel,
    'c1': d.c1,
    'c4': d.c4,
    'c5': d.c5,
    'mode': d.mode,
}

fuel_type = "jetA"
far_s, LHV = fuel_props(fuel_type)

# =========================================================
# EGR CASES (all at 75% load, 112 rpm)
# Vary: mf_tot, p_in, T_in, m_wiebe, phi_sc, phi_cd, Twalls
# EGR rate is passed via equ_in (fraction of recirculated exhaust)
# =========================================================

egr_cases = {
    0:  dict(mf_tot=0.034, p_in=3.65e5, T_in=390.0,
             m_wiebe=1.2, phi_sc=(363/180)*np.pi, phi_cd=(29/180)*np.pi,
             Twalls=[450, 450, 450]),
    10: dict(mf_tot=0.034, p_in=3.65e5, T_in=390.0,
             m_wiebe=1.2, phi_sc=(363/180)*np.pi, phi_cd=(29/180)*np.pi,
             Twalls=[450, 450, 450]),
    20: dict(mf_tot=0.034, p_in=3.65e5, T_in=390.0,
             m_wiebe=1.2, phi_sc=(363/180)*np.pi, phi_cd=(29/180)*np.pi,
             Twalls=[450, 450, 450]),
    30: dict(mf_tot=0.034, p_in=3.65e5, T_in=390.0,
             m_wiebe=1.2, phi_sc=(363/180)*np.pi, phi_cd=(29/180)*np.pi,
             Twalls=[450, 450, 450]),
    40: dict(mf_tot=0.034, p_in=3.65e5, T_in=390.0,
             m_wiebe=1.2, phi_sc=(363/180)*np.pi, phi_cd=(29/180)*np.pi,
             Twalls=[450, 450, 450]),
}
# =========================================================

egr_rates = [0, 10, 20, 30, 40]
results = {}

# --- Run simulations ---
for egr in egr_rates:
    lc = egr_cases[egr]
    print(f"\n--- Running EGR={egr}% | p_in={lc['p_in']*1e-5:.2f} bar | T_in={lc['T_in']} K ---")

    # EGR rate
    x_EGR = egr / 100.0

    #iterate on the correct equ_in based on a given EGR rate
    max_iter = 10
    tol = 1e-4

    #first guess on far exhaust
    far_exhaust = 0.03

    # add sweep to flag so nox is not calculated
    flags = ["sweep", "fuel_mass"]

    for i in range(max_iter):
        far_exhaust_old = far_exhaust  # save previous iteration value
        
        
        far_in = (x_EGR * far_exhaust) / (1 + far_exhaust * (1 - x_EGR))
        equ_in = far_in / far_s

        piston_input["mf_tot"]  = lc["mf_tot"]
        piston_input["p_in"]    = lc["p_in"]
        piston_input["T_in"]    = lc["T_in"]
        piston_input["m_wiebe"] = lc["m_wiebe"]
        piston_input["phi_sc"]  = lc["phi_sc"]
        piston_input["phi_cd"]  = lc["phi_cd"]
        piston_input["Twalls"]  = lc["Twalls"]
        piston_input["equ_in"]  = equ_in

        start = timer()
        piston_output = run_piston_engine(piston_input, flags)
        end = timer()
        print(f"  Time: {end - start:.2f} s")
        far_exhaust = piston_output["far exhaust"]
        print(f"far in: {far_in} and far_exhaust: {far_exhaust}")

        
        if abs(far_exhaust - far_exhaust_old) < tol:
                # run simulation once again without sweep in flags
                # to calculate NOX
                flags = ["fuel_mass"]
                piston_output = run_piston_engine(piston_input, flags)
                print(f"Converged in {i+1} iterations")
                break
    else:
        # the for-else clause triggers only if break was never hit
        print(f"Warning: did not converge after {max_iter} iterations")

  
    ca          = piston_output["crank angle trace"] * 360 / (2 * np.pi) - 180
    fuel_flow   = piston_output["fuel flow"]
    break_power = piston_output["break power"]
    bsfc        = (fuel_flow / break_power) * (1000 * 1000 * 3600)

    results[egr] = {
        "ca":          ca,
        "p_trace":     piston_output["pressure trace"],
        "T_trace":     piston_output["temperature trace"],
        "gross_heat":  piston_output["gross heat release"],
        "nox_rate":    piston_output["NO mass trace"],  
        "nox_times":   piston_output["NO times"],  
        "nox_angles":  piston_output["NO angles"] * 360 / (2 * np.pi) - 180,  
        "T_out":       piston_output["T_out"] - 272.15,
        "p_tdc":       piston_output["p_tdc"],
        "p_max":       piston_output["peak pressure"],
        "break_power": break_power,
        "bsfc":        bsfc,
        "nox_spec":    piston_output["nox_spec"] * 1.53,
        "equ_in":      equ_in, 
    }

    print(f"  T_out        : {results[egr]['T_out']:.1f} C")
    print(f"  p_tdc        : {results[egr]['p_tdc']*1e-5:.2f} bar")
    print(f"  p_max        : {results[egr]['p_max']*1e-5:.2f} bar")
    print(f"  Break power  : {break_power*1e-3:.2f} kW")
    print(f"  BSFC         : {bsfc:.1f} g/kWh")
    print(f"  NOx          : {results[egr]['nox_spec']:.4f} g/kWh")


# --- Helpers ---
def apply_minor(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# ── colours: one per EGR rate ─────────────────────────────
colors = plt.cm.viridis(np.linspace(0, 0.85, len(egr_rates)))

# --- Load experimental temperature trace data ---
exp_temp = {}
for egr in egr_rates:
    fpath = os.path.join("T_val", f"{egr}egr.txt")
    if os.path.exists(fpath):
        exp_temp[egr] = np.loadtxt(fpath, delimiter=",")
    else:
        exp_temp[egr] = None

# --- Load experimental cum heat release trace data ---
exp_heat = {}
for egr in egr_rates:
    fpath = os.path.join("heat_val_egr", f"{egr}egr.txt")
    if os.path.exists(fpath):
        exp_heat[egr] = np.loadtxt(fpath, delimiter=",")
    else:
        exp_heat[egr] = None


# --- Load experimental nox trace data ---
exp_nox = {}
for egr in egr_rates:
    fpath = os.path.join("nox_val", f"{egr}egr.txt")
    if os.path.exists(fpath):
        exp_nox[egr] = np.loadtxt(fpath, delimiter=",")
    else:
        exp_nox[egr] = None

# --- Load experimental KPI data ---
exp_kpi = {}
for name in ["bsfc", "nox"]:
    fpath = os.path.join("kpi_val_egr", f"{name}.txt")
    if os.path.exists(fpath):
        exp_kpi[name] = np.loadtxt(fpath, delimiter=",")
    else:
        exp_kpi[name] = None


# ── Fig 1: Pressure traces ────────────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
fig1.suptitle("Cylinder Pressure Traces — EGR Sweep (75% Load)")
axes1.flat[-1].set_visible(False)   # 5 cases, hide 6th panel
for ax, egr, col in zip(axes1.flat, egr_rates, colors):
    r = results[egr]
    ax.plot(r["ca"], r["p_trace"] * 1e-5, color=col, label="Calculated")
    ax.set_title(f"EGR = {egr}%")
    ax.set_xlim(120, 280)
    ax.set_ylim(0, 200)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    apply_minor(ax)
    ax.legend(fontsize=8)
for ax in axes1[1, :]: ax.set_xlabel("Crank Angle Degrees")
for ax in axes1[:, 0]: ax.set_ylabel("Cylinder Pressure [bar]")
fig1.tight_layout()


# ── Fig 2: Temperature traces ─────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle("Cylinder Temperature Traces — EGR Sweep (75% Load)")
axes2.flat[-1].set_visible(False)
for ax, egr, col in zip(axes2.flat, egr_rates, colors):
    r = results[egr]
    ax.plot(r["ca"], r["T_trace"], color=col, label="Calculated")
    if exp_temp[egr] is not None:
        ax.plot(exp_temp[egr][:, 0], exp_temp[egr][:, 1], "k--", label="Experimental")
    ax.set_title(f"EGR = {egr}%")
    ax.set_xlim(120, 280)
    apply_minor(ax)
    ax.legend(fontsize=8)
for ax in axes2[1, :]: ax.set_xlabel("Crank Angle Degrees")
for ax in axes2[:, 0]: ax.set_ylabel("Temperature [K]")
fig2.tight_layout()


# ── Fig 3: Gross heat release ─────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
fig3.suptitle("Cumulative Gross Heat Release — EGR Sweep (75% Load)")
axes3.flat[-1].set_visible(False)
for ax, egr, col in zip(axes3.flat, egr_rates, colors):
    r = results[egr]
    ax.plot(r["ca"], r["gross_heat"] * 1e-3, color=col, label="Calculated")
    if exp_heat[egr] is not None:
            ax.plot(exp_heat[egr][:, 0], exp_heat[egr][:, 1], "k--", label="Experimental")
    ax.set_title(f"EGR = {egr}%")
    ax.set_xlim(180, 220)
    ax.set_ylim(0, 1500)
    ax.yaxis.set_major_locator(MultipleLocator(400))
    apply_minor(ax)
    ax.legend(fontsize=8)
for ax in axes3[1, :]: ax.set_xlabel("Crank Angle Degrees")
for ax in axes3[:, 0]: ax.set_ylabel("Gross Heat Release [kJ]")
fig3.tight_layout()


# ── Fig 4: NOx formation rate ─────────────────────────────
fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
fig4.suptitle("NOx Formation Rate — EGR Sweep (75% Load)")
axes4.flat[-1].set_visible(False)
for ax, egr, col in zip(axes4.flat, egr_rates, colors):
    r = results[egr]
    ax.plot(r["nox_angles"], r["nox_rate"] * 1e6 * 1.53, color=col, label="Calculated") #convert to mg and from NO to NOx
    if exp_nox[egr] is not None:
            ax.plot(exp_nox[egr][:, 0], exp_nox[egr][:, 1], "k--", label="Experimental")
    ax.set_title(f"EGR = {egr}%")
    ax.set_ylim(0, 5000)
    ax.set_xlim(185, 215)
    apply_minor(ax)
    ax.legend(fontsize=8)
for ax in axes4[1, :]: ax.set_xlabel("Crank Angle Degrees")
for ax in axes4[:, 0]: ax.set_ylabel("Cylinder NOx formation [mg]")
fig4.tight_layout()


# ── Fig 5: Scalar KPIs vs EGR rate ───────────────────────
scalar_keys   = ["T_out",                  "p_tdc",      "p_max",      "break_power",     "bsfc",        "nox_spec"]
scalar_labels = ["Outlet Temperature [C]", "p_tdc [bar]","p_max [bar]","Break Power [kW]","BSFC [g/kWh]","NOx [g/kWh]"]
scalar_scales = [1,                         1e-5,         1e-5,         1e-3,              1,             1]
exp_kpi_keys  = [None,                  None,      None,      None,              "bsfc",        "nox"]

fig5, axes5 = plt.subplots(2, 3, figsize=(14, 8))
fig5.suptitle("Engine Performance vs EGR Rate (75% Load)")
for ax, key, label, scale, exp_key in zip(axes5.flat, scalar_keys, scalar_labels, scalar_scales, exp_kpi_keys):
    values = [results[egr][key] * scale for egr in egr_rates]
    ax.plot(egr_rates, values, marker="o", label="Calculated")
    if exp_key is not None and exp_kpi.get(exp_key) is not None:
        ax.plot(exp_kpi[exp_key][:, 0], exp_kpi[exp_key][:, 1], "k--", marker="^", label="Experimental")
    ax.set_xlabel("EGR Rate [%]")
    ax.set_ylabel(label)
    ax.set_xticks(egr_rates)
    ax.legend(fontsize=8)
    apply_minor(ax)
fig5.tight_layout()

plt.show()