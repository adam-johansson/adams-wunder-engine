import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Load data ---
seed = 14
pareto_df = pd.read_csv(f"optimisation_data/seed_{seed}/pareto_solutions.csv")
pareto_sorted = pareto_df.sort_values('eta_th')

x = pareto_sorted['eta_th'].values * 100
y = pareto_sorted['specific_nox'].values

x_fit = np.linspace(x.min(), x.max(), 200)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# --- Polynomial fit ---
poly_degree = 3  # try 2 or 4 too -- higher degrees risk overfitting/oscillation (Runge's phenomenon)
poly_coeffs = np.polyfit(x, y, poly_degree)
y_poly_fit = np.polyval(poly_coeffs, x_fit)
r2_poly = r_squared(y, np.polyval(poly_coeffs, x))
print(f"Polynomial (degree {poly_degree}) coefficients: {poly_coeffs}")
print(f"Polynomial R²: {r2_poly:.4f}")


# --- Exponential fit: y = a * exp(b * x) + c ---
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


# curve_fit needs a reasonable initial guess (p0) to converge -- adjust if it fails
p0 = [1.0, 0.05, 0.0]
try:
    exp_params, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
    y_exp_fit = exp_func(x_fit, *exp_params)
    r2_exp = r_squared(y, exp_func(x, *exp_params))
    print(f"Exponential fit: a={exp_params[0]:.4f}, b={exp_params[1]:.4f}, c={exp_params[2]:.4f}")
    print(f"Exponential R²: {r2_exp:.4f}")
except RuntimeError as e:
    print(f"Exponential fit failed to converge: {e}")
    print("Try adjusting the initial guess p0 -- e.g. a smaller/larger 'a', or flip the sign of 'b'.")
    y_exp_fit = None

# --- Plot for visual comparison ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, color='black', s=40, label='Pareto front data', zorder=5)
ax.plot(x_fit, y_poly_fit, color='tab:blue', linewidth=2, label=f'Polynomial fit (deg {poly_degree})')
if y_exp_fit is not None:
    ax.plot(x_fit, y_exp_fit, color='tab:red', linewidth=2, label='Exponential fit')

ax.set_xlabel(r"$\eta_{th}$ [%]")
ax.set_ylabel("Thrust specific NOx [mg/Ns]")
ax.legend()
ax.grid(True, color='lightgrey')
fig.tight_layout()
fig.savefig("curve_fit_comparison.pdf", dpi=300)
plt.show()