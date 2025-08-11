import torch
import torch.nn as nn

import numpy as np

from thermo import mixture_vectorized, JETA_L_vectorized, H2_vectorized, fuel_props


class PhysicsInformedLoss_vectorized(nn.Module):
    """
    Custom loss function that combines data loss with physics constraints
    """

    def __init__(self, x_scaler_params, y_scaler_params, fuel_type, data_weight=1.0, physics_weight=0.1):
        super(PhysicsInformedLoss_vectorized, self).__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()
        self.fuel_type = fuel_type

        # Store scaling parameters for inverse transformation
        self.x_mean = torch.tensor(x_scaler_params['mean'], dtype=torch.float64)
        self.x_std = torch.tensor(x_scaler_params['std'], dtype=torch.float64)
        self.y_mean = torch.tensor(y_scaler_params['mean'], dtype=torch.float64)
        self.y_std = torch.tensor(y_scaler_params['std'], dtype=torch.float64)

        # Fuel properties
        self.far_stoich, self.LHV = self._get_fuel_props(fuel_type)

    def _get_fuel_props(self, fuel_type):
        """Get stoichiometric fuel-air ratio and lower heating value"""

        far_stoich, LHV = fuel_props(fuel_type)

        return far_stoich, LHV

    def inverse_scale_inputs(self, scaled_inputs):
        """Convert scaled inputs back to physical units"""
        return scaled_inputs * self.x_std + self.x_mean

    def inverse_scale_outputs(self, scaled_outputs):
        """Convert scaled outputs back to physical units"""
        return scaled_outputs * self.y_std + self.y_mean

    def energy_conservation_constraint(self, inputs, outputs):
        """
        Enforce energy conservation in the combustion engine

        Energy balance:
        Energy_in = Energy_out + Power + Heat_loss

        Where:
        - Energy_in = mass_flow * cp * T_in + fuel_energy
        - Energy_out = mass_flow * cp * T_out
        - Power = mechanical power output
        - Heat_loss = heat loss to surroundings
        """
        # Convert scaled values back to physical units
        inputs_physical = self.inverse_scale_inputs(inputs)
        outputs_physical = self.inverse_scale_outputs(outputs)

        # Extract relevant variables (now in physical units)
        # Inputs: ['p_in', 'T_in', 'PI', 'cr', 'bore', 'v_mean', 'T_fuel', 'far']
        T_in = inputs_physical[:, 1]  # Temperature in (K)
        T_fuel = inputs_physical[:, 6]  # Fuel temperature (K)
        far = inputs_physical[:, 7]  # Fuel-air ratio (dimensionless)
        p_in = inputs_physical[:, 0]  # Inlet pressure
        p_ratio = inputs_physical[:, 2]
        p_out = p_ratio * p_in  # Outlet pressure

        # Outputs: ['power', 'heat_loss', 'nox', 'p_tdc', 'p_max', 'T_max', 'T_out', 'air_flow']
        power = outputs_physical[:, 0]  # Mechanical power (W)
        heat_loss = outputs_physical[:, 1]  # Heat loss (W)
        T_out = outputs_physical[:, 6]  # Temperature out (K)
        air_flow = outputs_physical[:, 7]  # Air mass flow (kg/s)

        equ_in = torch.zeros_like(T_in)

        # Vectorized calculation of specific enthalpy in, out and fuel enthalpy
        h_in, _, _, _, _, _, _, _ = mixture_vectorized(T_in, p_in, equ_in, fuel_type=self.fuel_type)
        h_out, _, _, _, _, _, _, _ = mixture_vectorized(T_out, p_out, far / self.far_stoich, fuel_type=self.fuel_type)

        if self.fuel_type == "jetA":
            _, h_fuel, _, _ = JETA_L_vectorized(T_fuel)
        else:
            _, h_fuel, _, _, _ = H2_vectorized(T_fuel, p_in)

        # Enthalpy in, out and fuel
        H_in = h_in * air_flow
        H_fuel = h_fuel * air_flow * far
        H_out = h_out * air_flow * (1 + far)

        # Conservation of energy
        energy_balance = H_in + H_fuel - H_out - power - heat_loss

        # normalise energy conservation with LHV
        fuel_energy_in = far * air_flow * self.LHV
        energy_balance_normalised = energy_balance / fuel_energy_in

        # Return squared constraint violation (should be close to zero)
        return torch.mean(energy_balance_normalised ** 2)

    def forward(self, predictions, targets, inputs):
        # Standard data fitting loss
        data_loss = self.mse_loss(predictions, targets)

        # Physics constraint - energy conservation only
        physics_loss = self.energy_conservation_constraint(inputs, predictions)

        # Combined loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss

        return total_loss, data_loss, physics_loss