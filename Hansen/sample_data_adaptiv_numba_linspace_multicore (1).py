import numpy as np
import importlib
import os
import shutil
import tempfile
from timeit import default_timer as timer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from piston_engine.engine import run_piston_engine  # import the piston engine function
import pandas as pd
from thermo import fuel_props

# Helper worker function to process a single simulation point
def run_single_simulation(idx, sample_point, piston_input_base, flags, csv_path, all_headers, n_out, lock, counter, total_points, start_idx, start_simulating, root_dir):
    if idx < start_idx:
        return 0  

    p, T, cr, bore, far_goal, p_ratio, v_mean = sample_point
    old_cwd = os.getcwd()
    removed = 0
    
    # Use tempfile.TemporaryDirectory as a context manager for guaranteed, automatic cleanup
    with tempfile.TemporaryDirectory(dir=root_dir, prefix=f"worker_{idx}_") as run_dir:
        try:
            # Copy necessary library files from the root directory into the managed temp directory
            for lib_file in ["thermo.lib", "trans.lib"]:
                src_path = os.path.join(root_dir, lib_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(run_dir, lib_file))
            
            # Move execution context into the unique folder
            os.chdir(run_dir)
            
            # Copy base input dict and modify for this specific run
            local_input = piston_input_base.copy()
            local_input.update({
                'p_in': p,
                'T_in': T,
                'p_ratio': p_ratio,
                'cr': cr,
                'bore': bore,
                'v_mean': v_mean,
                'lv_max': 0.1 * bore,
                'far_goal': far_goal,
            })

            # Run the engine simulation entirely inside this isolated folder
            piston_output = run_piston_engine(local_input, flags)

            try:
                T_out = piston_output["T_out"]   
            except (IndexError, KeyError):
                current_y = [0.0] * n_out
                removed = 1
            else:
                current_y = [
                    T_out,
                    piston_output["eta_th"],
                    piston_output["air_flow"],
                    piston_output["peak pressure"],
                    piston_output["peak temperature"],
                    piston_output["indicated power"],
                    piston_output["heat_loss"],
                    piston_output["p_tdc"],
                    piston_output["no_ppm"],
                    piston_output["flame temperature"],
                    piston_output["T start of combustion"],
                    piston_output["p start of combustion"],
                ]

            row_data = [p, T, cr, bore, far_goal, p_ratio, v_mean] + list(current_y)
            df_row = pd.DataFrame([row_data], columns=all_headers)

            # Thread/Process safe appending to the global CSV file
            with lock:
                write_header = not os.path.exists(csv_path)
                df_row.to_csv(csv_path, mode='a', index=False, header=write_header)
                
                # Increment global completed counter safely within lock
                counter.value += 1
                current_count = counter.value

                # Periodic status updates
                interval = total_points // 10 if total_points >= 10 else 1
                if current_count % interval == 0 or current_count == total_points:
                    mellantid = timer()
                    elapsed_time = mellantid - start_simulating
                    processed_this_session = current_count - start_idx
                    
                    avg_iteration_time = (elapsed_time / processed_this_session) if processed_this_session > 0 else 0
                    remaining_iterations = total_points - current_count
                    
                    print(
                        f"\n--- Parallel Progress Update ---"
                        f"\nCompleted {current_count} out of {total_points}."
                        f"\nElapsed time this session: {elapsed_time:.2f} [s]"
                        f"\nAvg iteration time (per point equivalent): {avg_iteration_time:.2f} [s]"
                        f"\nEstimated remaining session execution time: {(avg_iteration_time * remaining_iterations):.2f} [s]"
                        f"\nProgress in percent: {(current_count / total_points) * 100:.2f}%"
                        f"\n--------------------------------\n", flush=True
                    )

        finally:
            # Always revert to original working directory before context manager attempts deletion
            os.chdir(old_cwd)

    return removed


def main():
    # Capture absolute root directory before anything else changes
    root_dir = os.path.abspath(os.getcwd())

    # Setting up the piston engine
    input_file = "4stroke_sampling"

    if input_file == "4stroke_hydrogen_sampling":
        fuel = "H2"
    else:
        fuel = "jetA"

    input_dir = "piston_engine.input"
    path = input_dir + "." + input_file

    d = importlib.import_module(path)

    piston_input = {
        'cycle': d.cycle,
        'cooling': d.cooling,
        'opposed': d.opposed,
        'bsr': d.bsr,
        'lms': d.lms,
        'Twalls': d.Twalls,
        'ch': d.ch,
        'valve_timings': d.valve_timings,
        'n_valve': d.n_valve,
        'cd': d.cd,
        'eta_c': d.eta_c,
        'mf_tot': d.mf_tot,
        'wa': d.wa,
        'wm': d.wm,
        'm_wiebe': d.m_wiebe,
        'phi_sc': d.phi_sc,
        'phi_cd': d.phi_cd,
        'T_fuel': d.T_fuel,
        'p_fuel': d.p_fuel,
        'it': d.it,
        'wiebe_type': d.wiebe_type,
        'valve_type': d.valve_type,
        'cylinders': d.cylinders,
        'fuel': d.fuel,
        'c1': d.c1,
        'c4': d.c4,
        'c5': d.c5,
        'premixed': d.premixed,
        'mode': d.mode,
    }

    flags = ["sweep"]
    n_out = 12  

    start_sampling = timer()

    # ---------------------------------------------------------------
    # LINSPACE & CONSTANT CONFIGURATION SETTING
    # ---------------------------------------------------------------
    npoints = 100  # Number of steps for your linspace sweep
    """
    p_lim = [1e5, 35e5]  
    T_lim = [250, 1000]  
    cr_lim = [4, 16]  
    d_lim = [0.10, 0.20]  
    p_ratio_lim = [0.9, 1.5]
    v_mean_lim = [8, 15]
    far_lim = [0.02, 0.06]; JetA
    """

    # Choose which parameter to vary: "p_in", "T_in", "cr", "bore", "far_goal", "PI", or "v_mean"
    varying_param = "cr" 
    varying_range = [4, 16]  # [Min, Max] limits for the chosen parameter

    far_s, _ = fuel_props(fuel)
    default_far = (far_s / 2.2) if fuel == "jetA" else (far_s / 2.0)

    # Set default values for all parameters when they are kept constant
    constants = {
        "p_in": 1e5,
        "T_in": 600,
        "cr": 5,
        "bore": 0.1,
        "far_goal": default_far,
        "PI": 1.5,
        "v_mean": 11.5
    }
    # ---------------------------------------------------------------

    # Generate the linear space array for the selected variable
    linspace_array = np.linspace(varying_range[0], varying_range[1], npoints)
    
    # Construct the sequential execution matrix
    sample_scaled = np.zeros((npoints, 7))
    param_order = ["p_in", "T_in", "cr", "bore", "far_goal", "PI", "v_mean"]
    
    for idx, param in enumerate(param_order):
        if param == varying_param:
            sample_scaled[:, idx] = linspace_array
        else:
            sample_scaled[:, idx] = constants[param]

    headers_input = ["p_in", "T_in", "cr", "bore", "far_goal", "PI", "v_mean"]
    headers_output = ["T_out", "eff", "mdot_in", "p_max", "T_max", "power", "heat_loss", "p_tdc", "nox_ppm", "flame_temp", "T_sc", "p_sc"]
    all_headers = headers_input + headers_output

    # Calculate absolute CSV path before workers start changing their directories
    output_dir = os.path.join(root_dir, "piston_engine", "sampled_data", fuel)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{varying_param} - data.csv")

    start_idx = 0
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            start_idx = len(existing_df)
            print(f"Found existing CSV file. Resuming simulation from step {start_idx + 1}...")
        except Exception as e:
            print(f"Error reading existing CSV, starting from scratch. Error: {e}")
            start_idx = 0

    print(f"Total number of sampling points: {npoints}")
    start_simulating = timer()

    # Set up Multiprocessing Shared variables
    manager = multiprocessing.Manager()
    csv_lock = manager.Lock()
    global_counter = manager.Value('i', start_idx) 
    total_removed = 0

    # Leverage ProcessPoolExecutor for parallel core utility
    max_workers = max(os.cpu_count()-2, 2)
    print(f"Spawning pool utilizing {max_workers} CPU cores with isolated auto-cleaning workspaces...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample_point in enumerate(sample_scaled):
            if idx < start_idx:
                continue
            
            futures.append(
                executor.submit(
                    run_single_simulation,
                    idx,
                    sample_point,
                    piston_input,
                    flags,
                    csv_path,
                    all_headers,
                    n_out,
                    csv_lock,
                    global_counter,
                    npoints,
                    start_idx,
                    start_simulating,
                    root_dir
                )
            )

        for future in as_completed(futures):
            try:
                total_removed += future.result()
            except Exception as exc:
                print(f'A simulation generation step generated an exception: {exc}')

    end_sampling = timer()
    print(f"\nSimulation batch complete! Datapoints failed/removed this session: {total_removed}")
    print(f"Total execution time for this session: {end_sampling - start_sampling:.2f} [s]")


if __name__ == "__main__":
    main()