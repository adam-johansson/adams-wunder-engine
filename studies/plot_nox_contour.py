import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

print("Creating plots from saved CSV data...")

# Load both CSV datasets
df1 = pd.read_csv('nox_contour_data/jetA_nn.csv')
df2 = pd.read_csv('nox_contour_data/jetA_simulation.csv')

unique_lambdas1 = sorted(df1['lambda'].unique())
unique_lambdas2 = sorted(df2['lambda'].unique())

print(f"Found data for lambda values in dataset 1: {unique_lambdas1}")
print(f"Found data for lambda values in dataset 2: {unique_lambdas2}")

# Use the union of lambda values to ensure we plot all available data
all_lambdas = sorted(list(set(unique_lambdas1 + unique_lambdas2)))
print(f"Plotting lambda values: {all_lambdas}")

# Define output variables to plot
output_vars = {
    'NOx_g_per_kg': {'label': 'NOX [g/kg]', 'cmap': 'viridis'},
    'NOx_ppm': {'label': 'NOX [ppm]', 'cmap': 'plasma'},
    'air_flow': {'label': 'Air Flow', 'cmap': 'coolwarm'},
    'p_max': {'label': 'p_max [Pa]', 'cmap': 'inferno'},
    'T_max': {'label': 'T_max [K]', 'cmap': 'hot'},
    'T_out': {'label': 'T_out [K]', 'cmap': 'magma'}
}


def load_contour_from_csv(df, lambda_val, output_var):
    """Load contour data for a specific lambda value and output variable from CSV"""
    lambda_data = df[df['lambda'] == lambda_val].copy()

    if lambda_data.empty or output_var not in lambda_data.columns:
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
    Z_reshaped = lambda_data[output_var].values.reshape(grid_shape)

    return X_reshaped, Y_reshaped, Z_reshaped


def interpolate_simulation1_at_points(df1, df2, lambda_val, output_var):
    """Interpolate simulation 1 values at simulation 2 data points for comparison"""
    # Get simulation 2 points for this lambda
    sim2_data = df2[df2['lambda'] == lambda_val].copy()
    sim1_data = df1[df1['lambda'] == lambda_val].copy()

    if sim2_data.empty or sim1_data.empty or output_var not in sim2_data.columns:
        return None

    # Interpolate simulation 1 values at simulation 2 points
    sim1_interpolated = griddata(
        points=(sim1_data['p_in_Pa'], sim1_data['T_in_K']),
        values=sim1_data[output_var],
        xi=(sim2_data['p_in_Pa'], sim2_data['T_in_K']),
        method='linear'
    )

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'p_in_Pa': sim2_data['p_in_Pa'],
        'T_in_K': sim2_data['T_in_K'],
        'p_in_bar': sim2_data['p_in_Pa'] * 1e-5,
        'sim1_value': sim1_interpolated,
        'sim2_value': sim2_data[output_var].values,
        'error': sim1_interpolated - sim2_data[output_var].values,
        'relative_error': ((sim1_interpolated - sim2_data[output_var].values) / sim2_data[output_var].values) * 100
    })

    return comparison_df


# Create plots for each lambda value
for lambda_val in all_lambdas:
    # Create subplots: 3 rows (sim1, sim2, comparison), columns for output variables
    n_outputs = len(output_vars)
    fig, axes = plt.subplots(3, n_outputs, figsize=(4 * n_outputs, 12))

    # Handle case where there's only one output variable
    if n_outputs == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f'λ = {lambda_val}', fontsize=16, y=0.98)

    for col_idx, (output_var, var_info) in enumerate(output_vars.items()):
        # Plot simulation 1 (top row)
        ax1 = axes[0, col_idx]
        X_plot1, Y_plot1, Z_plot1 = load_contour_from_csv(df1, lambda_val, output_var)

        if X_plot1 is not None:
            contour1 = ax1.contourf(X_plot1 * 1e-5, Y_plot1, Z_plot1,
                                    levels=20, cmap=var_info['cmap'])
            ax1.set_title(f"Simulation 1 - {var_info['label']}")
            plt.colorbar(contour1, ax=ax1, label=var_info['label'])

            # Mark simulation 2 points on simulation 1 plot
            sim2_points = df2[df2['lambda'] == lambda_val]
            if not sim2_points.empty:
                ax1.scatter(sim2_points['p_in_Pa'] * 1e-5, sim2_points['T_in_K'],
                            c='red', s=30, marker='x', linewidths=2,
                            label='Sim2 points')
                ax1.legend()
        else:
            ax1.text(0.5, 0.5, f'No data available',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f"Simulation 1 - {var_info['label']}")

        ax1.set_xlabel('p_in (bar)')
        ax1.set_ylabel('T_in (K)')
        ax1.grid(True, alpha=0.3)

        # Plot simulation 2 (middle row) - scatter plot since only 25 points
        ax2 = axes[1, col_idx]
        sim2_data = df2[df2['lambda'] == lambda_val]

        if not sim2_data.empty and output_var in sim2_data.columns:
            scatter2 = ax2.scatter(sim2_data['p_in_Pa'] * 1e-5, sim2_data['T_in_K'],
                                   c=sim2_data[output_var], s=100, cmap=var_info['cmap'],
                                   edgecolors='black', linewidth=1)
            ax2.set_title(f"Simulation 2 - {var_info['label']} (25 points)")
            plt.colorbar(scatter2, ax=ax2, label=var_info['label'])
        else:
            ax2.text(0.5, 0.5, f'No data available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f"Simulation 2 - {var_info['label']}")

        ax2.set_xlabel('p_in (bar)')
        ax2.set_ylabel('T_in (K)')
        ax2.grid(True, alpha=0.3)

        # Plot comparison (bottom row) - error at simulation 2 points
        ax3 = axes[2, col_idx]
        comparison_df = interpolate_simulation1_at_points(df1, df2, lambda_val, output_var)

        if comparison_df is not None:
            # Remove NaN values for plotting
            valid_data = comparison_df.dropna()
            if not valid_data.empty:
                scatter3 = ax3.scatter(valid_data['p_in_bar'], valid_data['T_in_K'],
                                       c=valid_data['error'], s=100, cmap='RdBu_r',
                                       edgecolors='black', linewidth=1)
                ax3.set_title(f"Error (Sim1 - Sim2) - {var_info['label']}")
                cbar3 = plt.colorbar(scatter3, ax=ax3)
                cbar3.set_label(f'Error ({var_info["label"]})')

                # Add statistics
                rmse = np.sqrt(np.mean(valid_data['error'] ** 2))
                mae = np.mean(np.abs(valid_data['error']))
                ax3.text(0.02, 0.98, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
                         transform=ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, f'No comparison data',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f"Error - {var_info['label']}")

        ax3.set_xlabel('p_in (bar)')
        ax3.set_ylabel('T_in (K)')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save individual plots for each lambda
    filename_base = f'contour_plots_lambda_{lambda_val}'
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename_base}.pdf', bbox_inches='tight')
    print(f"Plots for λ = {lambda_val} saved as '{filename_base}.png' and '.pdf'")

    plt.show()

    # Print comparison statistics for each output variable
    print(f"\n=== Comparison Statistics for λ = {lambda_val} ===")
    for output_var, var_info in output_vars.items():
        comparison_df = interpolate_simulation1_at_points(df1, df2, lambda_val, output_var)
        if comparison_df is not None:
            valid_data = comparison_df.dropna()
            if not valid_data.empty:
                rmse = np.sqrt(np.mean(valid_data['error'] ** 2))
                mae = np.mean(np.abs(valid_data['error']))
                mean_rel_error = np.mean(np.abs(valid_data['relative_error']))
                print(f"{var_info['label']}:")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  Mean Relative Error: {mean_rel_error:.2f}%")

print("\nAll plots generated successfully!")