import numpy as np
import CoolProp.CoolProp as CP
import time

def FrontalA_FlowDirr(HEX):
	# ----------------------------------Set frontal area and HEX.flow----------------------------------
	# f1 frontal area
	if HEX.f1_flowdir=='Lx':
		f1_flow = 1
		HEX.f1_Afr = HEX.Ly*HEX.Lz
	elif HEX.f1_flowdir=='-Lx':
		f1_flow = -1
		HEX.f1_Afr = HEX.Ly*HEX.Lz
	elif HEX.f1_flowdir=='Ly':
		f1_flow = 2
		HEX.f1_Afr = HEX.Lx*HEX.Lz
	elif HEX.f1_flowdir=='-Ly':
		f1_flow = -2
		HEX.f1_Afr = HEX.Lx*HEX.Lz
	elif HEX.f1_flowdir=='Lz':
		f1_flow = 3
		HEX.f1_Afr = HEX.Lx*HEX.Ly
	elif HEX.f1_flowdir=='-Lz':
		f1_flow = -3
		HEX.f1_Afr = HEX.Lx*HEX.Ly
	else:
		print("Wrong flow direction given for f1")

	# f2 frontal area
	if HEX.f2_flowdir=='Lx':
		f2_flow = 1
		HEX.f2_Afr = HEX.Ly*HEX.Lz
	elif HEX.f2_flowdir=='-Lx':
		f2_flow = -1
		HEX.f2_Afr = HEX.Ly*HEX.Lz
	elif HEX.f2_flowdir=='Ly':
		f2_flow = 2
		HEX.f2_Afr = HEX.Lx*HEX.Lz
	elif HEX.f2_flowdir=='-Ly':
		f2_flow = -2
		HEX.f2_Afr = HEX.Lx*HEX.Lz
	elif HEX.f2_flowdir=='Lz':
		f2_flow = 3
		HEX.f2_Afr = HEX.Lx*HEX.Ly
	elif HEX.f2_flowdir=='-Lz':
		f2_flow = -3
		HEX.f2_Afr = HEX.Lx*HEX.Ly
	else:
		print("Wrong flow direction given for f2")

	if f1_flow==f2_flow:
		HEX.flow = "Parallel"
	elif f1_flow==-f2_flow:
		HEX.flow = "Counter"
	else:
		HEX.flow = "Cross"

	return HEX

def InitiateConditions(f, HEX):
	if not hasattr(f, 'rho'):
		f.rho = [CP.PropsSI("D", "P", f.p01, "T", f.T01, f.fluid)]
		f.h = [CP.PropsSI("H", "P", f.p01, "T", f.T01, f.fluid)]
		f.T0 = [f.T01]
		f.p0 = [f.p01]
		f.mu_mean = [np.array([1])]
		f.Pr_mean = [np.array([1])]
		f.k_mean = [np.array([1])]
		f.Re_mean = [np.array([1])]
	else:
		f.mu_mean.append(f.mu_mean[0])
		f.Pr_mean.append(f.Pr_mean[0])
		f.k_mean.append(f.k_mean[0])
		f.Re_mean.append(f.Re_mean[0])

	f.p0.append(f.p0[-1]*0.99)
	f.T0.append(f.T0[-1]*0.99)
	f.rho.append(f.rho[-1]*0.99)
	f.h.append(f.h[-1]*0.99)

	return f


def GetProperties_single(f, dataSource='Coolprop', verbose=False):
	f.T0_mean = (f.T0[-1]+f.T0[-2])/2
	f.p0_mean = (f.p0[-1]+f.p0[-2])/2

	if (dataSource == 'LOCAL') and (f.fluid == 'REFPROP::Parahydrogen'):
		if verbose:
			print("Local parahydrogen")
		fun = np.array([1/(f.T0[-1]**2), 1/(f.p0[-1]**2), 1/f.T0[-1], 1/f.p0[-1], 1*f.T0[-1], 1*f.p0[-1], 1*f.T0[-1]**2, 1*f.p0[-1]**2, 1], dtype=object)
		f.rho[-1] = np.sum(np.array([ 1.24437565e+05,  1.00000000e+00, -2.06794367e+03,  1.30246311e+05, -1.07976849e-01,  2.17380161e-06,  1.40972492e-04, -4.93692919e-14, 1.96131765e+01])*fun)
		f.h[-1] = np.sum(np.array([-2.69721272e+09,  1.00000000e+00,  1.13820931e+08,  1.00000000e+00, 2.38296543e+04, -9.44562022e-03, -1.31559159e+01,  6.24372971e-10, -1.79914578e+06])*fun)
		fun_mean = np.array([1/(f.T0_mean**2), 1/(f.p0_mean**2), 1/f.T0_mean, 1/f.p0_mean, 1*f.T0_mean, 1*f.p0_mean, 1*f.T0_mean**2, 1*f.p0_mean**2, 1], dtype=object)
		f.k_mean[-1] = np.sum(np.array([ 7.19288552e+02, 1.00000000e+00, -2.42108893e+01, -7.03004207e+03, -1.44089332e-04, 9.16771507e-10, 4.18344407e-07, 6.20207010e-17, 2.70227407e-01])*fun_mean)
		f.mu_mean[-1] = np.sum(np.array([ 1.60914156e-02, -3.11720433e+02, -4.42397409e-04, 4.53068470e-01, 1.10598934e-08, 1.24409540e-13, 9.71203227e-12, -2.47157542e-21, 5.44021040e-06])*fun_mean)
		f.Pr_mean[-1] = np.sum(np.array([ 4.27829545e+03, 1.00000000e+00, -9.91063663e+01, 4.06171091e+02, -3.19779186e-03, 1.64491717e-08, 3.97430618e-06, -1.33871848e-15, 1.52228725e+00])*fun_mean)

	elif (dataSource == 'LOCAL') and (f.fluid == 'Air'):
		if verbose:
			print("Local air")
		fun = np.array([1/(f.T0[-1]**2), 1/(f.p0[-1]**2), 1/f.T0[-1], 1/f.p0[-1], 1*f.T0[-1], 1*f.p0[-1], 1*f.T0[-1]**2, 1*f.p0[-1]**2, 1], dtype=object)
		f.rho[-1] = np.sum(np.array([ 2.52014191e+04,  1.00000000e+00,  1.46884379e+03,  7.05211266e+01, -7.26884160e-04,  9.16816372e-06,  3.85477344e-07,  2.89719551e-16, -3.82303508e+00])*fun)
		f.h[-1] = np.sum(np.array([8.08125770e+08, 1.00000000e+00, -1.23015129e+07, 5.57592228e+04, 8.45864957e+02, -1.34209705e-03, 1.49504691e-01, 1.83983609e-11, 1.90896315e+05])*fun)
		fun_mean = np.array([1/(f.T0_mean**2), 1/(f.p0_mean**2), 1/f.T0_mean, 1/f.p0_mean, 1*f.T0_mean, 1*f.p0_mean, 1*f.T0_mean**2, 1*f.p0_mean**2, 1], dtype=object)
		f.k_mean[-1] = np.sum(np.array([ 1.43556803e+02,  2.34416762e+03, -2.19648482e+00, -1.08149068e-02, 6.61908199e-05,  2.39443465e-10, -1.02436466e-08,  1.37574036e-17, 1.31794616e-02])*fun_mean)
		f.mu_mean[-1] = np.sum(np.array([ 1.11964165e-01, -1.54440082e-01, -1.78903379e-03,  1.43044853e-06, 4.17577120e-08,  1.18161457e-13, -8.78518790e-12,  4.71869288e-21, 1.15173755e-05])*fun_mean)
		f.Pr_mean[-1] = np.sum(np.array([-2.69467615e+03, -2.22223327e+04,  4.20663962e+01,  4.84828161e-01, 7.41433297e-05,  5.21905954e-09,  7.15333520e-08, -1.69928019e-16, 5.68654081e-01])*fun)

	elif (dataSource == 'Coolprop_incomp'):
		f.rho[-1] = np.array([CP.PropsSI("D", "P", f.p0[-1], "T", f.T0[-1], f.fluid)])
		f.rho_mean = (f.rho[-1] + f.rho[-2]) / 2
		f.h[-1] = CP.PropsSI("H", "P", f.p0_mean, "T", f.T0[-1], f.fluid)
		f.mu_mean[-1] = CP.PropsSI("V", "P", f.p0_mean, "T", f.T0_mean, f.fluid)
		f.Pr_mean[-1] = CP.PropsSI("PRANDTL", "P", f.p0_mean, "T", f.T0_mean, f.fluid)
		f.k_mean[-1] = CP.PropsSI("CONDUCTIVITY", "P", f.p0_mean, "T", f.T0_mean, f.fluid)

	else:
		if verbose:
			print("Coolprop")
		# print('p0:',f.p0,'T0:', f.T0)
		f.rho[-1] = np.array([CP.PropsSI("D", "P", f.p0[-1], "T", f.T0[-1], f.fluid)])
		f.rho_mean = (f.rho[-1]+f.rho[-2])/2
		f.h[-1] = CP.PropsSI("H", "D", f.rho[-1], "T", f.T0[-1], f.fluid)
		f.mu_mean[-1] = CP.PropsSI("V", "D", f.rho_mean, "T", f.T0_mean, f.fluid)
		f.Pr_mean[-1] = CP.PropsSI("PRANDTL", "D", f.rho_mean, "T", f.T0_mean, f.fluid)
		f.k_mean[-1] = CP.PropsSI("CONDUCTIVITY", "D", f.rho_mean, "T", f.T0_mean, f.fluid)

	return f


def KL_curvefit(HEX):
	# f1 side
	HEX.f1_l_d_h = HEX.f1_L/(4*HEX.f1_sigma/HEX.f1_alpha)
	k1, a, b, c = [ 3.60039451e-01, -4.00821276e-01,  2.12762744e-05, -4.12945363e-01]
	HEX.f1_j = k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h
	HEX.f1_St = HEX.f1_j/(HEX.f1_Pr**(2/3))
	HEX.f1_Nu = HEX.f1_St*HEX.f1_Re*HEX.f1_Pr # = HEX.f1_j*HEX.Re*HEX.Pr**(1/3)

	k1, a, b, c = [ 5.08290733e-01, -5.51216597e-01,  1.09255635e-04, -2.34092870e-01]

	HEX.f1_f = k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h

	# f2 side
	HEX.f2_l_d_h = HEX.f2_L/(4*HEX.f2_sigma/HEX.f2_alpha)
	k1, a, b, c = [3.60039451e-01, -4.00821276e-01, 2.12762744e-05, -4.12945363e-01]
	HEX.f2_j = k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h
	HEX.f2_St = HEX.f2_j/(HEX.f2_Pr**(2/3))
	HEX.f2_Nu = HEX.f2_St*HEX.f2_Re*HEX.f2_Pr

	k1, a, b, c = [5.08290733e-01, -5.51216597e-01, 1.09255635e-04, -2.34092870e-01]

	HEX.f2_f = k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h

	return HEX

def GenHex_single(f1, f2, HEX, verbose=True):

	# ----------------------------------Set frontal area and HEX.flow----------------------------------
	HEX = FrontalA_FlowDirr(HEX)

	# ----------------------------------Calculations outside iteration----------------------------------
	HEX.f2_sigma = (1-HEX.chi)/(1+HEX.sigma_r)
	HEX.f1_sigma = HEX.f2_sigma*HEX.sigma_r

	HEX.f2_alpha = 2/HEX.wt*HEX.chi/(HEX.alpha_r+1)
	HEX.f1_alpha = HEX.f2_alpha*HEX.alpha_r

	HEX.f1_sigma_ftt = (HEX.f1_alpha-HEX.f2_alpha)/HEX.f1_alpha
	HEX.f2_sigma_ftt = (HEX.f2_alpha-HEX.f1_alpha)/HEX.f2_alpha

	# ----------------------------------Initial conditions----------------------------------
	f1 = InitiateConditions(f1, HEX)
	f2 = InitiateConditions(f2, HEX)

	# ----------------------------------Aerothermal iterative loop start----------------------------------
	# (iterating density and enthalpy for mean conditions)
	t = time.time()
	numIter = 10
	for i in range(numIter):
		if verbose:
			print("Aerothermal iteration",i,"after",time.time()-t,"s")
			t_i = time.time()
		# ----------------------------------Get f1 conditions----------------------------------
		f1 = GetProperties_single(f1, dataSource=f1.dataSource, verbose=verbose)
		if verbose:
			print("Got all f1 values from CoolProp in",time.time()-t_i,"s")
			t_i = time.time()

		# ----------------------------------Get f2 conditions----------------------------------
		f2 = GetProperties_single(f2, dataSource=f2.dataSource, verbose=verbose)
		if verbose:
			print("Got all f2 values from CoolProp in",time.time()-t_i,"s")
			t_i = time.time()

		# ----------------------------------Calculations----------------------------------
		# Reynolds number
		f1.Re_mean[-1] = 4*f1.mdot/(HEX.f1_alpha*HEX.f1_Afr*f1.mu_mean[-1])
		f2.Re_mean[-1] = 4*f2.mdot/(HEX.f2_alpha*HEX.f2_Afr*f2.mu_mean[-1])

		HEX.f1_Re = f1.Re_mean[-1]
		HEX.f1_Pr = f1.Pr_mean[-1]

		HEX.f2_Re = f2.Re_mean[-1]
		HEX.f2_Pr = f2.Pr_mean[-1]

		# Baseline heat transfer and pressure loss coefficients
		if HEX.correlation == "Gnielinski":
			# Assume both sides as internal of tubes
			f1 = gnielinskiBlended(f1)
			f2 = gnielinskiBlended(f2)
		elif HEX.correlation == "Lienhard":
			# Assume both sides as flat plates
			f1 = Lienhard(f1, HEX.Lx)
			f2 = Lienhard(f2, HEX.Ly)
		elif HEX.correlation == "KaysAndLondon":
			HEX = KaysAndLondonHEX(HEX, f1, f2)
		else:
			# Both sides from Kays and London curve-fit correlation
			# f1 = KL_curvefit(f1)
			# f2 = KL_curvefit(f2)
			HEX = KL_curvefit(HEX)

		# Nusselt to heat transfer coefficient
		HEX.f1_h = HEX.f1_Nu*f1.k_mean[-1]/(4*HEX.f1_sigma/HEX.f1_alpha)
		HEX.f2_h = HEX.f2_Nu*f2.k_mean[-1]/(4*HEX.f2_sigma/HEX.f2_alpha)

		if verbose:
			print("Calculated heat transfer coefficient in",time.time()-t_i,"s")
			t_i = time.time()

		# Assemble U_f1
		HEX.f1_h_ave = HEX.f1_h
		HEX.f1_f_ave = HEX.f1_f

		HEX.f2_h_ave = HEX.f2_h
		HEX.f2_f_ave = HEX.f2_f

		# -----------Where f1 is enhanced side-----------
		HEX.f1_eta_O = 1
		HEX.f2_eta_O = 1
		if HEX.f1_sigma_ftt >= 0:

			# Fin efficiency
			HEX.f1_ml = HEX.FinAR*(2*HEX.f1_h/HEX.k)**0.5
			HEX.f1_eta_fin = np.tanh(HEX.f1_ml)/HEX.f1_ml

			# Overall surface efficiency
			# HEX.f1_eta_O = np.ones_like(HEX.f1_sigma)
			HEX.f1_eta_O = (1-HEX.f1_sigma_ftt*(1-HEX.f1_eta_fin))
		# -----------Where f2 is enhanced side-----------
		elif HEX.f2_sigma_ftt>=0:

			# Fin efficiency
			HEX.f2_ml = HEX.FinAR*(2*HEX.f2_h/HEX.k)**0.5
			HEX.f2_eta_fin = np.tanh(HEX.f2_ml)/HEX.f2_ml

			# Overall surface efficiency
			# HEX.f2_eta_O = np.ones_like(HEX.f2_sigma)
			HEX.f2_eta_O = (1-HEX.f2_sigma_ftt*(1-HEX.f2_eta_fin))
		# -----------overall conductance (f1 side)-----------
		HEX.U_f1 = 1/(1/(HEX.f1_eta_O*HEX.f1_h_ave)+HEX.wt/((HEX.f1_alpha+HEX.f2_alpha)/(2*HEX.f1_alpha)*HEX.k)+1/(HEX.f2_eta_O*HEX.f2_h_ave*HEX.f2_alpha/HEX.f1_alpha))

		# -----------Calc temp change (effectivness based on f1 side)-----------
		# Fluid heat capacity rate
		HEX.f1_C = (f1.h[-1]-f1.h[-2])/(f1.T0[-1]-f1.T0[-2])*f1.mdot
		HEX.f2_C = (f2.h[-1]-f2.h[-2])/(f2.T0[-1]-f2.T0[-2])*f2.mdot

		# Capacity rate min and ratio
		HEX.Cmin = np.minimum(HEX.f1_C, HEX.f2_C)
		HEX.Cr = HEX.Cmin/np.maximum(HEX.f1_C, HEX.f2_C)

		# Maximum number of transfer units
		HEX.NTUmax = HEX.Lx*HEX.Ly*HEX.Lz*HEX.f1_alpha*HEX.U_f1/HEX.Cmin
		if verbose:
			print("Calculated NTU in",time.time()-t_i,"s")
			t_i = time.time()

		# Effectivness
		if HEX.flow == "Counter":
			# For counter flow
			HEX.eps = (1-np.exp(-HEX.NTUmax*(1-HEX.Cr)))/(1-HEX.Cr*np.exp(-HEX.NTUmax*(1-HEX.Cr)))
		elif HEX.flow=="Parallel":
			print("No equation for parallel flow given yet, going with cross flow")
			HEX.eps = (1-np.exp(-HEX.NTUmax*(1+HEX.Cr)))/(1+HEX.Cr)
		elif HEX.flow == "Cross":
			# For cross flow, both fluid unmixed
			# HEX.eps = 1-np.exp((1/HEX.Cr)*HEX.NTUmax**.22*(np.exp(-HEX.Cr*HEX.NTUmax**.78)-1))
			# For cross flow, both fluid mixed
			HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.Cr*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.Cr))-1)
		else:

			HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.Cr*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.Cr))-1)

		# Adjust effectivness for longitudinal conduction (hard to do from Kays, but not really needed)
		#HEX.eps = HEX.eps*(1-)

		f1_dT0 = -HEX.eps*(HEX.Cmin/HEX.f1_C)*(f1.T0[-2]-f2.T0[-2])
		f2_dT0 = HEX.eps*(HEX.Cmin/HEX.f2_C)*(f1.T0[-2]-f2.T0[-2])

		# -----------Calculate pressure loss-----------
		f1_dp0 = - 0.5*f1.mdot**2/(f1.rho[-2])*1/(HEX.f1_Afr)**2*((1/HEX.f1_sigma**2+1)*(f1.rho[-2]/f1.rho[
			-1]-1)+HEX.f1_f_ave*HEX.Lx*HEX.Ly*HEX.Lz/HEX.f1_Afr*HEX.f1_alpha/HEX.f1_sigma**3*2*f1.rho[-2]/(
																			  f1.rho[-2]+f1.rho[-1]))
		f2_dp0 = - 0.5*f2.mdot**2/(f2.rho[-2])*1/(HEX.f2_Afr)**2*((1/HEX.f2_sigma**2+1)*(f2.rho[-2]/f2.rho[
			-1]-1)+HEX.f2_f_ave*HEX.Lx*HEX.Ly*HEX.Lz/HEX.f2_Afr*HEX.f2_alpha/HEX.f2_sigma**3*2*f2.rho[-2]/(
																			  f2.rho[-2]+f2.rho[-1]))

		# Check convergence
		# T0_f1
		if np.max(1-np.abs((f1.T0[-1]-f1.T0[-2])/f1_dT0))<0.001:
			# T0_f2
			if np.max(1-np.abs((f2.T0[-1]-f2.T0[-2])/f2_dT0))<0.001:
				# p0_f1
				if np.max(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1_dp0))<0.001:
					# p0_f2
					if np.max(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2_dp0))<0.001:
						f1.T0[-1] = f1.T0[-2]+f1_dT0
						f1.p0[-1] = f1.p0[-2]+f1_dp0
						f2.T0[-1] = f2.T0[-2]+f2_dT0
						f2.p0[-1] = f2.p0[-2]+f2_dp0  #print(np.mean(f1.T0[-1,...]), np.mean(f2.T0[-1,...]))
				break

		if i==numIter-1:
			print("Did not converge")
			print("Max dT0_1", np.max(f1.T0[-1]-f1.T0[-2]))
			print("Max dT0_2", np.max(f2.T0[-1]-f2.T0[-2]))
			print("Max dp0_1", np.max(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1.p0[-2])))
			print("Max dp0_2", np.max(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2.p0[-2])))

		f1.T0[-1] = f1.T0[-2]+f1_dT0
		f1.p0[-1] = f1.p0[-2]+f1_dp0
		f2.T0[-1] = f2.T0[-2]+f2_dT0
		f2.p0[-1] = f2.p0[-2]+f2_dp0

		if verbose:
			print("Calculated dT0 and dp0 in", time.time()-t_i, "s")
			t_i = time.time()

		HEX.mass = HEX.Lx*HEX.Ly*HEX.Lz*HEX.chi*HEX.density  # Alu

	return f1, f2, HEX

