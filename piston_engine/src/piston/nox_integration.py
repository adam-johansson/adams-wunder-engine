import numpy as np
from scipy.integrate import solve_ivp
import warnings


def solve_nox_ode_robust(dNOdt_fun, times, initial_conditions=None):
    """
    Robust ODE solver specifically designed for stiff NOx chemistry.

    Handles the characteristic behavior of NOx formation:
    - Long periods of slow chemistry (small derivatives)
    - Sudden bursts of fast chemistry during high-T combustion (large derivatives)

    Parameters
    ----------
    dNOdt_fun : callable
        ODE function f(t, y) returning dy/dt
    times : array_like
        Time points where solution is needed
    initial_conditions : array_like, optional
        Initial conditions. Defaults to [0.0] for NOx

    Returns
    -------
    sol : OdeResult
        Solution object from scipy
    """

    if initial_conditions is None:
        initial_conditions = np.array([0.0])

    t_span = (times[0], times[-1])

    # Strategy 1: Try implicit method first (best for stiff problems)
    methods_to_try = [
        {
            'method': 'Radau',  # Implicit method, excellent for stiff problems
            'rtol': 1e-6,  # Reasonable relative tolerance
            'atol': 1e-12,  # Very small absolute tolerance for small NOx concentrations
            'max_step': np.inf,  # Let solver choose step size
            'description': 'Radau (implicit, stiff-stable)'
        },
        {
            'method': 'BDF',  # Backward differentiation - also good for stiff
            'rtol': 1e-6,
            'atol': 1e-12,
            'max_step': np.inf,
            'description': 'BDF (implicit, multistep)'
        },
        {
            'method': 'LSODA',  # Adaptive method that switches between stiff/non-stiff
            'rtol': 1e-6,
            'atol': 1e-12,
            'max_step': np.inf,
            'description': 'LSODA (adaptive stiff/non-stiff)'
        },
        {
            'method': 'RK45',  # Fallback explicit method with tight tolerances
            'rtol': 1e-8,
            'atol': 1e-14,
            'max_step': 1e-6,  # Very small steps for stability
            'description': 'RK45 (explicit, tight tolerances)'
        }
    ]

    for solver_config in methods_to_try:
        try:
            print(f"Trying {solver_config['description']}...")

            sol = solve_ivp(
                dNOdt_fun,
                t_span,
                initial_conditions,
                method=solver_config['method'],
                t_eval=times,
                rtol=solver_config['rtol'],
                atol=solver_config['atol'],
                max_step=solver_config['max_step'],
                dense_output=False,  # Save memory
            )

            if sol.success:
                print(f"✓ Success with {solver_config['description']}")
                print(f"  Function evaluations: {sol.nfev}")
                print(f"  Integration steps: {len(sol.t) if hasattr(sol, 't') else 'N/A'}")
                return sol
            else:
                print(f"✗ Failed: {sol.message}")

        except Exception as e:
            print(f"✗ Exception with {solver_config['description']}: {e}")
            continue

    # If all methods fail, raise an error
    raise RuntimeError("All ODE integration methods failed. Check your ODE function for issues.")


def solve_nox_ode_adaptive_stepping(dNOdt_fun, times, initial_conditions=None):
    """
    Alternative approach: Adaptive stepping based on chemistry activity.

    Uses dense time grid during high-temperature periods and sparse grid
    during low-activity periods.
    """

    if initial_conditions is None:
        initial_conditions = np.array([0.0])

    # Identify high-activity regions (you'd need to pass temperature data)
    # This is a simplified example - you'd adapt based on your specific case

    # Create adaptive time grid
    adaptive_times = _create_adaptive_time_grid(times)

    # Solve on dense grid
    sol_dense = solve_ivp(
        dNOdt_fun,
        (adaptive_times[0], adaptive_times[-1]),
        initial_conditions,
        method='Radau',  # Use stiff-stable method
        t_eval=adaptive_times,
        rtol=1e-6,
        atol=1e-12,
        max_step=np.inf,
    )

    if not sol_dense.success:
        raise RuntimeError(f"Adaptive stepping failed: {sol_dense.message}")

    # Interpolate back to original time grid
    sol_interp = _interpolate_solution(sol_dense, times)

    return sol_interp


def _create_adaptive_time_grid(times, refinement_factor=10):
    """
    Create denser time grid in regions where rapid changes are expected.

    In practice, you'd identify high-temperature regions and refine there.
    """

    # Simple example: refine middle portion (typically where combustion occurs)
    n_points = len(times)
    start_idx = n_points // 4
    end_idx = 3 * n_points // 4

    # Normal spacing outside combustion region
    t_early = times[:start_idx]
    t_late = times[end_idx:]

    # Dense spacing during combustion
    t_combustion_coarse = times[start_idx:end_idx]
    t_combustion_dense = np.linspace(
        t_combustion_coarse[0],
        t_combustion_coarse[-1],
        len(t_combustion_coarse) * refinement_factor
    )

    # Combine
    adaptive_times = np.concatenate([t_early, t_combustion_dense, t_late])
    return np.unique(adaptive_times)  # Remove duplicates and sort


def _interpolate_solution(sol_dense, target_times):
    """Interpolate dense solution to target time points."""
    from scipy.interpolate import interp1d

    # Create interpolation function
    interp_func = interp1d(
        sol_dense.t,
        sol_dense.y[0],
        kind='cubic',
        bounds_error=False,
        fill_value='extrapolate'
    )

    # Interpolate to target times
    y_interp = interp_func(target_times)

    # Create mock solution object
    class MockSolution:
        def __init__(self, t, y, success=True):
            self.t = t
            self.y = np.array([y])
            self.success = success
            self.message = "Interpolated from dense solution"

    return MockSolution(target_times, y_interp)


def diagnose_ode_stiffness(dNOdt_fun, times, y0=None):
    """
    Diagnostic function to analyze the stiffness of your NOx ODE.

    Helps understand why integration might be difficult.
    """

    if y0 is None:
        y0 = np.array([0.0])

    print("=== ODE Stiffness Diagnosis ===")

    # Sample the derivative at different time points
    n_samples = min(100, len(times))
    sample_indices = np.linspace(0, len(times) - 1, n_samples, dtype=int)
    sample_times = times[sample_indices]

    derivatives = []
    for t in sample_times:
        try:
            dydt = dNOdt_fun(t, y0)
            derivatives.append(abs(dydt[0]))
        except:
            derivatives.append(np.nan)

    derivatives = np.array(derivatives)
    valid_derivs = derivatives[~np.isnan(derivatives)]

    if len(valid_derivs) == 0:
        print("❌ Could not evaluate derivatives - check ODE function")
        return

    # Analyze derivative behavior
    min_deriv = np.min(valid_derivs)
    max_deriv = np.max(valid_derivs)
    ratio = max_deriv / min_deriv if min_deriv > 0 else np.inf

    print(f"Derivative range: {min_deriv:.2e} to {max_deriv:.2e}")
    print(f"Dynamic range ratio: {ratio:.2e}")

    # Stiffness indicators
    if ratio > 1e6:
        print("🔴 VERY STIFF: Use implicit methods (Radau, BDF)")
    elif ratio > 1e3:
        print("🟡 MODERATELY STIFF: Consider implicit methods or LSODA")
    else:
        print("🟢 NOT STIFF: RK45 should work fine")

    # Time scale analysis
    char_time = 1.0 / np.mean(valid_derivs) if np.mean(valid_derivs) > 0 else np.inf
    min_time_step = times[1] - times[0]

    print(f"Characteristic chemistry time: {char_time:.2e} s")
    print(f"Minimum time step: {min_time_step:.2e} s")

    if char_time < min_time_step * 100:
        print("⚠️  Chemistry is much faster than time resolution")
        print("   Consider: denser time grid or implicit methods")


# Example usage in your NOx calculation:
def improved_nox_integration(dNOdt_fun, times):
    """
    Drop-in replacement for your original ODE solving with much better
    handling of stiff NOx chemistry.
    """

    # First, diagnose the problem
    print("Analyzing NOx chemistry stiffness...")
    diagnose_ode_stiffness(dNOdt_fun, times)

    # Then solve with robust method
    try:
        sol = solve_nox_ode_robust(dNOdt_fun, times)
        print("✓ NOx integration completed successfully")
        return sol

    except RuntimeError as e:
        print(f"❌ Robust integration failed: {e}")
        print("Trying adaptive stepping approach...")

        try:
            sol = solve_nox_ode_adaptive_stepping(dNOdt_fun, times)
            print("✓ Adaptive stepping successful")
            return sol
        except Exception as e2:
            raise RuntimeError(f"All integration approaches failed. Original: {e}, Adaptive: {e2}")

