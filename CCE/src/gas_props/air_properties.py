
import numpy as np
#from src.misc import spline
from scipy import interpolate
import matplotlib.pyplot as plt

def isa(alt,disa,show):
    #path = r'src/properties/air_properties/'
    #data=np.loadtxt(path + 'ISA.txt', dtype=float)
    #alt_vec = data[:, 0]
    alt_vec = np.linspace(0,20000,41)
    #tvec = data[:, 1]
    tvec = np.asarray([288.15,284.9,281.65,278.4,275.15,271.9,268.65,265.4,262.15,258.9,255.65,252.4,249.15,245.9,242.65,239.4,
            236.15,232.9,229.65,226.4,223.15,219.9,216.65,216.65,216.65,216.65,216.65,216.65,216.65,216.65,216.65,216.65,
            216.65,216.65,216.65,216.65,216.65,216.65,216.65,216.65,216.65])
    #pvec = data[:, 2]
    pvec = np.asarray([101325,95460.8,89874.6,84556,79495.2,74682.5,70108.5,65764.1,61640.2,57728.3,54019.9,50506.8,47181,44034.8,
            41060.7,38251.4,35599.8,33099,30742.5,28523.6,26436.3,24474.4,22632.1,20980.0,19400.4,17864.8,16510.4,15258.7,
            14101.8,13032.7,12044.6,11131.4,10287.5,9507.5,8786.68,8120.51,7504.84,6935.86,6410.01,5924.03,5474.89])

    fp = interpolate.interp1d(alt_vec, pvec)
    ft = interpolate.interp1d(alt_vec, tvec)
    p = float(fp(alt))
    t = ft(alt) + disa
    c = np.sqrt(1.4 * t * 287)
    if show:
        plt.plot(alt_vec, tvec, 'o', alt, ft(alt))
        plt.show()

    return p, t, c

def isa_form(alt,disa):
    psl = 101325
    tsl = 288.15
    if alt >= 0 and alt < 11000.0:
        p = psl*(1-0.0065*alt/tsl)**5.2561
        t = tsl-6.5*alt/1000 +disa
    elif alt >= 11000.0 and alt <=20000:
        t = 216.65 + disa
        p = 22.63253/np.exp(0.000157689*(alt-10998.1))*1000

    else:
        print('altitudes above 20km not implemented')

    c = np.sqrt(1.4 * t * 287)
    return p,t,c

def get_k_air(t):
    if t < 100.0 or t > 1000.0:
        ValueError('temperature out of range in get_k_air')

    C1 = -2.917e-008
    C2 = 9.5e-005
    C3 = 0.0003102

    k = C1 * t ** 2 + C2 * t + C3

    return k


# dynamic viscosity
def get_mu_air(t):
    mu_0 = 17.1E-6

    if t > 273.15 and t < 373.15:
        beta1 = 0.4743742289
        beta2 = 0.0445219791
        beta3 = 0.1651857290
    elif t > 373.15 and t < 973.15:
        beta1 = 0.5688972460
        beta2 = 0.0414204258
        beta3 = 0.1192514508
    else:
        if t < 273.15:
            mu = 17.1E-6
        if t > 973.15:
            mu = 41.5E-6

        return mu

    mu = mu_0 * ((t / 288.15) ** beta1 + beta2 * (t / 288.15) + beta3 * np.log((t / 288.15) ** 2))

    return mu


# data for air at 1atm "fundamentals of heat and mass transfer"
def get_prandtl_air(t):
    t_vect = np.asarray([100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,
                         1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,3000])

    pr_vect = np.asarray([0.786,0.758,0.737,0.72,0.707,0.7,0.69,0.686,0.684,0.683,0.685,0.69,0.695,0.702,0.709,0.716,
                          0.72,0.723,0.726,0.728,0.728,0.719,0.703,0.685,0.688,0.685,0.683,0.677,0.672,0.667,0.655,
                          0.647,0.63,0.613,0.536,])

    if t < 100.0 or t > 3000.0:
        ValueError('temperature outside table range in get_prandtl')

    pr_out = spline(t_vect, pr_vect, t, 3, False, 'get_prandtl')


    return pr_out

def get_cp_air(t):
    t_vect = np.asarray([100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,
                         1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,3000])

    cp_vect=np.asarray([1.032,1.012,1.007,1.006,1.007,1.009,1.014,1.021,1.03,1.04,1.051,1.063,1.075,1.087,1.099,
                        1.11,1.121,1.131,1.141,1.159,1.175,1.189,1.207,1.23,1.248,1.267,1.286,1.307,1.337,1.372,
                        1.417,1.478,1.558,1.665,2.726])

    if t < 100.0 or t > 3000.0:
        ValueError('temperature outside table range in get_prandtl')

    cp_out = spline(t_vect, cp_vect, t, 3, False, 'get_cp_air')

    return(cp_out*1000)

