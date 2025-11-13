import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

import sys
sys.path.append("./../")

from thermo import fuel_props, JETA_L, H2, mixture, molar_fractions

from piston_engine.engine import run_piston_engine
