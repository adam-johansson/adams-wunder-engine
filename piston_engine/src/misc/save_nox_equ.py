import numpy as np





def save_nox(phi, evo, sc, no, no_equ, p, T):


    # high pressure crank angles
    phi_hp = phi[(phi > sc) & (phi < evo)]

    phi_hp = phi_hp[::100]
    no = no[::100]
    no_equ = no_equ[::100]
    p = p[::100]
    T = T[::100]
    
    no = np.vstack((phi_hp * 180 / np.pi, no)).transpose()
    no_equ = np.vstack((phi_hp * 180 / np.pi, no_equ)).transpose()
    p = np.vstack((phi_hp * 180 / np.pi, p*1e-5)).transpose()
    T = np.vstack((phi_hp * 180 / np.pi, T)).transpose()


    np.savetxt("piston_engine/simulation_data/NOX/no2.dat", no, fmt="%.5f")
    np.savetxt("piston_engine/simulation_data/NOX/no_equ2.dat", no_equ, fmt="%.5f")
    np.savetxt("piston_engine/simulation_data/NOX/p2.dat", p, fmt="%.5f")
    np.savetxt("piston_engine/simulation_data/NOX/T2.dat", T, fmt="%.5f")





    return