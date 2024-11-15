import numpy as np
from numba import jit


@jit(nopython=True)
def dmvdphi(h_abs,h_open,n_valve,cd,width_port,p,T,gamma,R,p3,T3,gamma3,R3,type):
    if h_abs > h_open:
        backflow = False
        if type =='out':
            if p/p3 > 1:
                p1 = p
                p2 = p3
                T1 = T
                R1 = R
                gamma1 = gamma
            else:
                p1 = p3
                p2 = p
                T1 = T3
                R1 = R3
                gamma1 = gamma3
                backflow = True
        elif type == 'in':
            if p3/p > 1:
                p1 = p3
                p2 = p
                T1 = T3
                R1 = R3
                gamma1 = gamma3
            else:
                p1 = p
                p2 = p3
                T1 = T
                gamma1 = gamma
                R1 = R
                backflow = True
        else:
            print('Unknown valve')
            return
        

        h = h_abs - h_open

        
        
        R = width_port/2
        A_max =  0.25 * np.pi * width_port**2 
        # if port is circular
        if h < R:
            A = ((R**2) * np.arccos(1-h/R) - (R-h)*np.sqrt(R**2 - (R - h)**2)) #*1.065
        elif h < width_port:
            h2 = width_port - h
            A = A_max - ((R**2) * np.arccos(1-h2/R) - (R-h2)*np.sqrt(R**2 - (R - h2)**2))
        else:
            A = A_max
        
        """
        
        h_max = width_port*0.48
        if h < h_max:
            A = width_port*h
        else:
            A = width_port*h_max
        """
        Aeff = A*cd*n_valve 
        
        
        PI = max(p2/p1, (2/(gamma1 + 1))**(gamma1/(gamma1-1)) )
        FF = np.sqrt((gamma1/(gamma1-1)) * (PI**(2/gamma1) -  PI**((gamma1+1)/gamma1)))
        

        if backflow:
            return -Aeff * np.sqrt(2/(R1*T1)) * p1 * FF
        else:
            return Aeff * np.sqrt(2/(R1*T1)) * p1 * FF
    
    
    
    return 0.0
