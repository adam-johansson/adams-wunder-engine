from thermo import polynomials


T = 298.15
p = 1e5

cp_h2, h_h2, s_h2, _, M_h2 = polynomials.H2(T, p)
cp_o2, h_o2, s_o2, _, M_o2 = polynomials.O2(T, p)
cp_h2o, h_h2o, s_h2o, _, M_h2o = polynomials.H2O(T, p)
cp_co2, h_co2, s_co2, _, M_co2 = polynomials.CO2(T, p)
cp_jetA, h_jetA, s_jetA, M_jetA = polynomials.JETA(T)

# M (molar mass) is kg/mol

# calculate the heating value of hydrogen (per kg mixture)

# 1 mole h2 + 0.5 mole O2

# output from polynomials is mass based (per kg)
# enthalpy has units kJ??

# mass of oxygen
m_o2 = 0.5 * M_o2

# mass of hydrogen
m_h2 = M_h2

# mass of water
m_h2o = M_h2o

# mass of reactants
m1 = m_o2 + m_h2

# mass of products
m2 = m_h2o

H1 = m_h2 * h_h2 + m_o2 * h_o2

H2 = m_h2o * h_h2o

# LHV
LHV = -(H2 - H1) / m_h2


print(f"Mass 1 H2 + O2 mixture = {m1}")
print(f"Mass 2 H2O= {m2}")
print(f"LHV of H2= {LHV*1e-6} MJ/kg")

# calculate the heating value of jetA (C_12H_23)

# 1 mole C12_H23

# output from polynomials is mass based (per kg)
# enthalpy has units J / kg

# mass of jetA
m_jetA = 1 * M_jetA

# mass of hydrogen
m_h2 = M_h2

# mass of oxygen
m_o2 = 17.75 * M_o2

# mass of reactants
m1 = m_o2 + m_jetA

# mass of CO2
m_CO2 = 12 * M_co2

# mass of H2O
m_h2o = 11.5 * M_h2o

# mass of products
m2 = m_h2o + m_CO2

# enthalpy of reactants
H1 = m_jetA * h_jetA + m_o2 * h_o2

# enthalpy of products
H2 = m_CO2 * h_co2 + m_h2o * h_h2o

# LHV
LHV = -(H2 - H1) / m_jetA


print(f"Mass 1 H2 + O2 mixture = {m1}")
print(f"Mass 2 CO2 + H2O = {m2}")
print(f"LHV of jetA = {LHV*1e-6} MJ/kg")
