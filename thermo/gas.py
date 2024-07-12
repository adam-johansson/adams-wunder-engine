import pandas as pd
import os
import math

R_uni = 8.3144626  # J mol^-1 K^-1  Universal gas constant
p_std = 1e5  # pressure at standard state


class PureSubstance:
    """Dataset Class to store the samples and their corresponding labels,
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    """

    # substance is e.g. N2, O2, CO2, H2O, Ar
    def __init__(self, substance, temperature=298.15, pressure=1e5):
        self.substance = substance
        self.temperature = temperature
        # partial pressure of the gas if the gas is in a mixture
        self.pressure = pressure
        self._coefficients = self._get_coefficients()
        self.molar_mass = self._coefficients.loc["molar_mass"]

    def _get_coefficients(self):
        script_dir = os.path.dirname(__file__)  # Get the script directory
        relative_path = 'coefficients.csv'  # Adjust this to your file's location
        full_path = os.path.join(script_dir, relative_path)
        coefficients = pd.read_csv(full_path, index_col=0).loc[self.substance]

        return coefficients

    def heat_capacity(self):

        if self.temperature < 200:
            T = 200
        elif self.temperature > 6000:
            T = 6000
        else:
            T = self.temperature

        R = R_uni / self.molar_mass  # J kg^-1 K^-1 specific gas constant

        if T < 1000.0007: #between 200K and 1000K
            a1 = self._coefficients.loc["a1l"]
            a2 = self._coefficients.loc["a2l"]
            a3 = self._coefficients.loc["a3l"]
            a4 = self._coefficients.loc["a4l"]
            a5 = self._coefficients.loc["a5l"]
            a6 = self._coefficients.loc["a6l"]
            a7 = self._coefficients.loc["a7l"]

        else:  # 1000K to 6000K
            a1 = self._coefficients.loc["a1h"]
            a2 = self._coefficients.loc["a2h"]
            a3 = self._coefficients.loc["a3h"]
            a4 = self._coefficients.loc["a4h"]
            a5 = self._coefficients.loc["a5h"]
            a6 = self._coefficients.loc["a6h"]
            a7 = self._coefficients.loc["a7h"]

        cp = R * (a1 * T ** (-2) + a2 * T ** (-1) + a3 + a4 * T + a5 * T ** 2 + a6 * T ** 3 + a7 * T ** 4)

        return cp

    def enthalpy(self):
        if self.temperature < 200:
            T = 200
        elif self.temperature > 6000:
            T = 6000
        else:
            T = self.temperature

        R = R_uni / self.molar_mass  # J kg^-1 K^-1 specific gas constant

        if T < 1000.0007: #between 200K and 1000K
            a1 = self._coefficients.loc["a1l"]
            a2 = self._coefficients.loc["a2l"]
            a3 = self._coefficients.loc["a3l"]
            a4 = self._coefficients.loc["a4l"]
            a5 = self._coefficients.loc["a5l"]
            a6 = self._coefficients.loc["a6l"]
            a7 = self._coefficients.loc["a7l"]
            b1 = self._coefficients.loc["b1l"]

        else:  # 1000K to 6000K
            a1 = self._coefficients.loc["a1h"]
            a2 = self._coefficients.loc["a2h"]
            a3 = self._coefficients.loc["a3h"]
            a4 = self._coefficients.loc["a4h"]
            a5 = self._coefficients.loc["a5h"]
            a6 = self._coefficients.loc["a6h"]
            a7 = self._coefficients.loc["a7h"]
            b1 = self._coefficients.loc["b1h"]

        h = R * T * (-a1 * T ** (-2) + a2 * math.log(T) / T + a3 + a4 * T / 2 + a5 * T ** 2 / 3 + a6 * T ** 3 / 4
                     + a7 * T ** 4 / 5 + b1 / T)

        # All reference elements have assigned enthalpy values equal to zero at 298.15K
        # Add H(298.15) - H(0) to set reference to 0K (doesn't change anything)
        h = h + self._coefficients.loc["0K_ref"] / self._coefficients.loc["molar_mass"]

        # add heat of formation

        h = h - self._coefficients.loc["heat_of_formation"] / self.molar_mass

        return h

    def entropy(self):
        if self.temperature < 200:
            T = 200
        elif self.temperature > 6000:
            T = 6000
        else:
            T = self.temperature

        R = R_uni / self.molar_mass # J kg^-1 K^-1 specific gas constant

        if T < 1000.0007: #between 200K and 1000K
            a1 = self._coefficients.loc["a1l"]
            a2 = self._coefficients.loc["a2l"]
            a3 = self._coefficients.loc["a3l"]
            a4 = self._coefficients.loc["a4l"]
            a5 = self._coefficients.loc["a5l"]
            a6 = self._coefficients.loc["a6l"]
            a7 = self._coefficients.loc["a7l"]
            b2 = self._coefficients.loc["b2l"]

        else:  # 1000K to 6000K
            a1 = self._coefficients.loc["a1h"]
            a2 = self._coefficients.loc["a2h"]
            a3 = self._coefficients.loc["a3h"]
            a4 = self._coefficients.loc["a4h"]
            a5 = self._coefficients.loc["a5h"]
            a6 = self._coefficients.loc["a6h"]
            a7 = self._coefficients.loc["a7h"]
            b2 = self._coefficients.loc["b2h"]

        s = R * (-a1 * T ** (-2) / 2 - a2 * T ** (-1) + a3 * math.log(T) + a4 * T + a5 * T ** 2 / 2 + a6 * T ** 3 / 3
                 + a7 * T ** 4 / 4 + b2)

        # pressure dependence of the entropy
        s = s - R * math.log(self.pressure / p_std)

        return s


