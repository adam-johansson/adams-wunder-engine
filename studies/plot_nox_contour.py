import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Creating plots from saved CSV data...")

# Load both CSV datasets
df1 = pd.read_csv('nox_contour_data.csv')
df2 = pd.read_csv('nox_contour_data_jetA_simulation.csv')

unique_lambdas1 = sorted(df1['lambda'].unique())
unique_lambdas2 = sorted(df2['lambda'].unique())

print(f"Found data for lambda values in dataset 1: {unique_lambdas1}")
print(f"Found data for lambda values in dataset 2: {unique_lambdas2}")

# Use the union of lambda values to ensure we plot all available data
all_lambdas = sorted(list(set(unique_lambdas1 + unique_lambdas2)))
print(f"Plotting lambda values: {all_lambdas}")


def load_contour_from_csv(df, lambda_val):
    """Load contour data for a specific lambda value from CSV"""
    lambda_data = df[df['lambda'] == lambda_val].copy()

    if lambda_data.empty:
        return None, None, None

    # Get unique pressure and temperature values to determine grid shape
    unique_p = sorted(lambda_data['p_in_Pa'].unique())
    unique_T = sorted(lambda_data['T_in_K'].unique())

    grid_shape = (len(unique_T), len(unique_p))  # Note: Y first, then X

    # Sort data by T_in_K first, then by p_in_Pa to ensure correct reshaping
    lambda_data = lambda_data.sort_values(['T_in_K', 'p_in_Pa'])

    # Reshape back to 2D arrays
    X_reshaped = lambda_data['p_in_Pa'].values.reshape(grid_shape)
    Y_reshaped = lambda_data['T_in_K'].values.reshape(grid_shape)
    Z_reshaped = lambda_data['NOx_g_per_kg'].values.reshape(grid_shape)

    return X_reshaped, Y_reshaped, Z_reshaped


# Create subplots: 2 rows (for each simulation), columns for lambda values
levels = 20
n_lambdas = len(all_lambdas)
fig, axes = plt.subplots(2, n_lambdas, figsize=(4 * n_lambdas, 8))

# Handle case where there's only one lambda value
if n_lambdas == 1:
    axes = axes.reshape(2, 1)

for col_idx, lambda_val in enumerate(all_lambdas):
    # Plot simulation 1 (top row)
    ax1 = axes[0, col_idx]
    X_plot1, Y_plot1, Z_plot1 = load_contour_from_csv(df1, lambda_val)

    if X_plot1 is not None:
        contour1 = ax1.contourf(X_plot1 * 1e-5, Y_plot1, Z_plot1,
                                levels=levels, cmap='viridis')
        ax1.set_title(f"Simulation 1 - λ = {lambda_val}")
        plt.colorbar(contour1, ax=ax1, label='NOX [g/kg]')
    else:
        ax1.text(0.5, 0.5, f'No data for λ = {lambda_val}',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f"Simulation 1 - λ = {lambda_val}")

    ax1.set_xlabel('p_in (bar)')
    ax1.set_ylabel('T_in (K)')
    ax1.grid(True, alpha=0.3)

    # Plot simulation 2 (bottom row)
    ax2 = axes[1, col_idx]
    X_plot2, Y_plot2, Z_plot2 = load_contour_from_csv(df2, lambda_val)

    if X_plot2 is not None:
        contour2 = ax2.contourf(X_plot2 * 1e-5, Y_plot2, Z_plot2,
                                levels=levels, cmap='viridis')
        ax2.set_title(f"Simulation 2 - λ = {lambda_val}")
        plt.colorbar(contour2, ax=ax2, label='NOX [g/kg]')
    else:
        ax2.text(0.5, 0.5, f'No data for λ = {lambda_val}',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f"Simulation 2 - λ = {lambda_val}")

    ax2.set_xlabel('p_in (bar)')
    ax2.set_ylabel('T_in (K)')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('nox_contour_plots_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('nox_contour_plots_comparison.pdf', bbox_inches='tight')
print("Comparison plots saved as 'nox_contour_plots_comparison.png' and '.pdf'")