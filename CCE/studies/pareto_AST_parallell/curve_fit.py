import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_pareto(seed):
    df = pd.read_csv(f"optimisation_data/seed_{seed}/pareto_solutions.csv")
    df = df.sort_values('eta_th')
    x = df['eta_th'].values * 100
    y = df['specific_nox'].values
    return x, y


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def fit_exponential(x, y, label, p0=(1.0, 0.05, 0.0)):
    params, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
    r2 = r_squared(y, exp_func(x, *params))
    print(f"[{label}] Exponential fit: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
    print(f"[{label}] Exponential R²: {r2:.4f}")
    return params


# --- Background Pareto front (seed 13) ---
x_ref, y_ref = load_pareto(13)

# --- Curves to fit and overlay ---
curves = [
    (15, "200 bar"),
    (19, "1350K"),
]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_ref, y_ref, color='black', s=40, label='Pareto front (seed 13)', zorder=5)

colors = ['tab:red', 'tab:blue']
for (seed, label), color in zip(curves, colors):
    x, y = load_pareto(seed)
    x_fit = np.linspace(x.min(), x.max(), 200)
    try:
        params = fit_exponential(x, y, label)
        y_fit = exp_func(x_fit, *params)
        ax.plot(x_fit, y_fit, color=color, linewidth=2, label=f'{label} (exponential fit)')
    except RuntimeError as e:
        print(f"[{label}] Exponential fit failed to converge: {e}")
        print("Try adjusting the initial guess p0 -- e.g. a smaller/larger 'a', or flip the sign of 'b'.")

ax.set_xlabel(r"$\eta_{th}$ [%]")
ax.set_ylabel("Thrust specific NOx [mg/Ns]")
ax.legend()
ax.grid(True, color='lightgrey')
fig.tight_layout()
fig.savefig("curve_fit_comparison.pdf", dpi=300)
plt.show()