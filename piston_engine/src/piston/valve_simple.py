def dmvdphi(phi, phi_open, phi_close,dtdphi,kv,p1,p2):
    #kv is the valve coef kg/s/Pa
    #dtdphi is the derivative of time with respect to crank angle
    if phi >= phi_open and phi <= phi_close:
        if p2<p1:
            return 0.0
        return kv*(p2-p1)*dtdphi
    return 0.0
