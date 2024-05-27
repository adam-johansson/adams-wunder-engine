from CCE.src import cce_propulsion_system
from CCE.src import auxiliaries
from scipy.optimize import differential_evolution, basinhopping, minimize
from CCE.src import misc

import os

import numpy as np
import matplotlib.pyplot as plt


def global_optimisation(data, data_piston, flags, meta_model):

    # delete csv file from old optimsation run
    # Python program to delete a csv file
    # csv file present in same directory

    # first check whether file exists or not
    # calling remove method to delete the csv file
    # in remove method you need to pass file name and type
    file = 'optimisation_data/evaluations.csv'
    if (os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)

    def global_sfc(x):

        #print(f'bpr: {x[0]}, opr: {x[1]}, split: {x[2]}, pi_pe: {x[3]}, cr: {x[4]}, bore: {x[5]}')
        bpr = x[0]  # bypass ratio
        opr = x[1]
        split = x[2]  # pressure ratio hpc
        pi_pe = x[3]  # piston engine pressure increase or drop
        cr = x[4]
        bore = x[5]


        data[2] = bpr
        #data[3] = T35
        data[31] = opr
        data[32] = split
        data[27] = pi_pe
        data[30] = cr
        data[33] = bore
        # maybe diameter? or match core flow?

        #print('ny iter')

        #sfc, v_ratio, F, m0, error = cce_propulsion_system.run_cce(data, flags)
        sfc, v_ratio, F, m0, error, fpr, p_max, T_max, T_in_piston, T_out_piston, TET, far_piston\
            = auxiliaries.run_cce_fpr(data, data_piston, flags, meta_model)
        #if error:
        #    #print('Not working power plant.')
        #    return 1e6

        # COST SHOULD BE SFC. BUT ALSO PENALISE SOME OTHER STUFF MAYBE

        cost = sfc * 1e6
        if cost < 5.0:
            print(f'bpr: {x[0]}, opr: {x[1]}, split: {x[2]}, pi_pe: {x[3]}, cr: {x[4]}, fpr: {fpr}, bore: {x[5]}')
            print(sfc, F)
            misc.optimisation_csv(sfc * 1e6, opr, split, pi_pe, cr, bore, fpr, bpr, p_max, T_max, T_in_piston,
                                  T_out_piston, TET, far_piston)
        return cost


    bpr_lim = [10, 45]
    #tet_lim = [1000, 1400]
    opr_lim = [5, 50]
    split_lim = [0.1, 0.9]
    pi_pe_lim = [0.9, 1.7]
    cr_lim = [4, 16]
    bore_lim = [0.05, 0.2]
    bounds = (bpr_lim, opr_lim, split_lim, pi_pe_lim, cr_lim, bore_lim)

    x0 = np.array([16.55, 19.985, 0.55, 1.21, 4.87, 0.098])

    opt = differential_evolution(global_sfc, bounds=bounds)
    #opt = minimize(global_sfc, x0, bounds=bounds)

    print(f'Optimal BPR: {opt.x[0]}')
    print(f'Optimal OPR: {opt.x[1]}')
    print(f'Optimal split: {opt.x[2]}')
    print(f'Optimal PR PE: {opt.x[3]}')
    print(f'Optimal compression ratio: {opt.x[4]}')
    print(f'Optimal bore: {opt.x[5]}')

    return
