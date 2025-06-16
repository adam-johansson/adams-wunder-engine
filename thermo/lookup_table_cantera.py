#!/usr/bin/env python3
"""
Cantera Lookup Table Generator

This script generates lookup tables for gas concentrations to speed up
ODE solver calculations by avoiding repeated Cantera equilibrium calculations.

Usage:
    python generate_lookup_tables.py
"""

import numpy as np
import pandas as pd
import cantera as ct
import pickle
import os
import time
from datetime import datetime

# Constants (adjust these to match your code)
R_UNIV_J = 8314.46261815324  # J/(kmol·K)
SPECIES_THRESHOLD = 1e-15


class LookupTableGenerator:
    def __init__(self, fuel_type, T_range, p_range, initial_fractions, output_dir="lookup_tables"):
        """
        Initialize the lookup table generator.

        Args:
            fuel_type: "jetA" or "H2"
            T_range: (T_min, T_max, T_points)
            p_range: (p_min, p_max, p_points)
            initial_fractions: tuple of initial species fractions
            output_dir: directory to save lookup tables
        """
        self.fuel_type = fuel_type
        self.T_min, self.T_max, self.T_points = T_range
        self.p_min, self.p_max, self.p_points = p_range
        self.initial_fractions = initial_fractions
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create grids
        self.T_grid = np.linspace(self.T_min, self.T_max, self.T_points)
        self.p_grid = np.linspace(self.p_min, self.p_max, self.p_points)

        # Setup Cantera gas object
        self.gas = self._setup_cantera_gas(fuel_type)

        print(f"Initialized lookup table generator for {fuel_type}")
        print(f"Temperature range: {self.T_min}-{self.T_max} K ({self.T_points} points)")
        print(f"Pressure range: {self.p_min / 1e5:.1f}-{self.p_max / 1e5:.1f} bar ({self.p_points} points)")
        print(f"Total calculations needed: {self.T_points * self.p_points}")

    def _setup_cantera_gas(self, fuel_type):
        """Setup Cantera gas object with appropriate species."""
        try:
            species = {S.name: S for S in ct.Species.list_from_file("gri30.yaml")}
        except Exception as e:
            raise RuntimeError(f"Failed to load Cantera species: {e}")

        if fuel_type == "jetA":
            species_names = ["CO2", "H2O", "O2", "CO", "OH", "H2", "O", "H", "N2"]
        elif fuel_type == "H2":
            species_names = ["H2O", "O2", "OH", "H2", "O", "H", "N2"]
        else:
            raise ValueError(f"Unsupported fuel type: {fuel_type}")

        try:
            ohc_species = [species[name] for name in species_names]
            gas = ct.Solution(thermo="ideal-gas", species=ohc_species)
            return gas
        except KeyError as e:
            raise RuntimeError(f"Missing species in Cantera database: {e}")

    def _get_fallback_concentrations(self, T, p):
        """Fallback concentrations when equilibration fails."""
        # Simple temperature-dependent fallback (adjust as needed)
        c_O = 1e-10 * np.exp(-15000 / T) * p / (R_UNIV_J * T)
        c_N2 = 0.78 * p / (R_UNIV_J * T)
        c_O2 = 0.21 * p / (R_UNIV_J * T)
        c_OH = 1e-8 * np.exp(-8000 / T) * p / (R_UNIV_J * T)
        c_H = 1e-12 * np.exp(-20000 / T) * p / (R_UNIV_J * T)
        return c_O, c_N2, c_O2, c_OH, c_H

    def _calculate_equilibrium(self, T, p):
        """Calculate equilibrium concentrations at given T and p."""
        # Unpack initial fractions
        (xi_N2_0, xi_CO2_0, xi_H2O_0, xi_CO_0, xi_O2_0,
         xi_OH_0, xi_H2_0, xi_O_0, xi_H_0) = self.initial_fractions

        # Set gas composition (without NO for lookup table)
        try:
            if self.fuel_type == "jetA":
                composition = (
                    f"CO2:{xi_CO2_0}, H2O:{xi_H2O_0}, O2:{xi_O2_0}, N2:{xi_N2_0}, "
                    f"CO:{xi_CO_0}, OH:{xi_OH_0}, H2:{xi_H2_0}, O:{xi_O_0}, H:{xi_H_0}"
                )
            else:  # H2
                composition = (
                    f"H2O:{xi_H2O_0}, O2:{xi_O2_0}, N2:{xi_N2_0}, "
                    f"OH:{xi_OH_0}, H2:{xi_H2_0}, O:{xi_O_0}, H:{xi_H_0}"
                )

            self.gas.TPX = T, p, composition
            self.gas.equilibrate("TP")

        except Exception as e:
            return self._get_fallback_concentrations(T, p)

        # Extract equilibrium concentrations
        fractions = self.gas.mole_fraction_dict(threshold=SPECIES_THRESHOLD)

        # Get species fractions with fallback to 0
        xi_O = fractions.get("O", 0.0)
        xi_H = fractions.get("H", 0.0)
        xi_O2 = fractions.get("O2", 0.0)
        xi_OH = fractions.get("OH", 0.0)
        xi_N2 = fractions.get("N2", 0.0)

        # Convert to concentrations
        c_H = (xi_H * p) / (R_UNIV_J * T)
        c_O2 = (xi_O2 * p) / (R_UNIV_J * T)
        c_O = (xi_O * p) / (R_UNIV_J * T)
        c_OH = (xi_OH * p) / (R_UNIV_J * T)
        c_N2 = (xi_N2 * p) / (R_UNIV_J * T)

        return c_O, c_N2, c_O2, c_OH, c_H

    def generate_tables(self, save_csv=True, save_pickle=True):
        """Generate lookup tables and save them."""
        print("\nStarting lookup table generation...")
        start_time = time.time()

        # Initialize result arrays
        results = []

        total_points = self.T_points * self.p_points
        successful_calcs = 0
        failed_calcs = 0

        for i, T in enumerate(self.T_grid):
            for j, p in enumerate(self.p_grid):
                calc_num = i * self.p_points + j + 1

                # Progress indicator
                if calc_num % max(1, total_points // 20) == 0:
                    elapsed = time.time() - start_time
                    progress = calc_num / total_points
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    print(f"Progress: {calc_num}/{total_points} ({100 * progress:.1f}%) - "
                          f"ETA: {eta / 60:.1f} min")

                try:
                    c_O, c_N2, c_O2, c_OH, c_H = self._calculate_equilibrium(T, p)
                    successful_calcs += 1
                    status = "success"
                except Exception as e:
                    c_O, c_N2, c_O2, c_OH, c_H = self._get_fallback_concentrations(T, p)
                    failed_calcs += 1
                    status = "fallback"

                # Store result
                results.append({
                    'T': T,
                    'p': p,
                    'c_O': c_O,
                    'c_N2': c_N2,
                    'c_O2': c_O2,
                    'c_OH': c_OH,
                    'c_H': c_H,
                    'status': status
                })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"cantera_lookup_{self.fuel_type}_{timestamp}"

        # Save as CSV
        if save_csv:
            csv_filename = os.path.join(self.output_dir, f"{base_filename}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"\nSaved CSV lookup table: {csv_filename}")

        # Save as pickle (includes metadata)
        if save_pickle:
            pickle_data = {
                'fuel_type': self.fuel_type,
                'T_range': (self.T_min, self.T_max, self.T_points),
                'p_range': (self.p_min, self.p_max, self.p_points),
                'initial_fractions': self.initial_fractions,
                'T_grid': self.T_grid,
                'p_grid': self.p_grid,
                'data': df,
                'metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'successful_calcs': successful_calcs,
                    'failed_calcs': failed_calcs,
                    'total_time_seconds': time.time() - start_time
                }
            }

            pickle_filename = os.path.join(self.output_dir, f"{base_filename}.pkl")
            with open(pickle_filename, 'wb') as f:
                pickle.dump(pickle_data, f)
            print(f"Saved pickle lookup table: {pickle_filename}")

        # Print summary
        total_time = time.time() - start_time
        print(f"\nGeneration complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Successful calculations: {successful_calcs}")
        print(f"Failed calculations: {failed_calcs}")
        print(f"Success rate: {100 * successful_calcs / total_points:.1f}%")

        return df


def main():
    """Main function to generate lookup tables."""

    # Configuration - ADJUST THESE VALUES FOR YOUR CASE
    fuel_type = "jetA"  # or "H2"

    # Temperature and pressure ranges (adjust based on your problem)
    T_range = (1000, 3000, 100)  # T_min, T_max, T_points
    p_range = (1e5, 10e5, 50)  # p_min, p_max, p_points

    # Initial fractions - ADJUST THESE TO MATCH YOUR PROBLEM
    # Order: (xi_N2_0, xi_CO2_0, xi_H2O_0, xi_CO_0, xi_O2_0, xi_OH_0, xi_H2_0, xi_O_0, xi_H_0)
    initial_fractions = (0.75, 0.10, 0.10, 0.01, 0.03, 0.005, 0.005, 1e-6, 1e-6)

    # Create generator
    generator = LookupTableGenerator(
        fuel_type=fuel_type,
        T_range=T_range,
        p_range=p_range,
        initial_fractions=initial_fractions
    )

    # Generate tables
    df = generator.generate_tables(save_csv=True, save_pickle=True)

    print(f"\nFirst few rows of generated data:")
    print(df.head())

    print(f"\nData statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()