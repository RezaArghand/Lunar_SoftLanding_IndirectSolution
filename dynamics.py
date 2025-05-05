import numpy as np

def dynamics(tau, Y, tf, params):
    """State and costate equations"""
    r, phi, u, v = Y[0], Y[1], Y[2], Y[3]
    lr, lphi, lu, lv = Y[4], Y[5], Y[6], Y[7]
    
    t = tau * tf
    m = max(params['M0'] - params['mdot'] * t, 10)
    beta = np.arctan2(lv, lu)
    
    # State equations
    dr = tf * v
    dphi = tf * u / r
    du = tf * (-u*v/r + params['T']*np.cos(beta)/m)
    dv = tf * (u**2/r - params['mu']/r**2 + params['T']*np.sin(beta)/m)
    
    # Costate equations
    dlr = tf * (lphi*u/r**2 - lu*u*v/r**2 - lv*(u**2/r**2 - 2*params['mu']/r**3))
    dlphi = 0
    dlu = tf * (-lphi/r + lu*v/r - 2*lv*u/r)
    dlv = tf * (-lr + lu*u/r)
    
    return np.array([dr, dphi, du, dv, dlr, dlphi, dlu, dlv])

def compute_hamiltonian(Y, tf, params):
    """Calculate Hamiltonian value"""
    r, u, v = Y[0], Y[2], Y[3]
    lr, lphi, lu, lv = Y[4], Y[5], Y[6], Y[7]
    
    m = max(params['M0'] - params['mdot'] * tf, 10)
    beta = np.arctan2(lv, lu)
    
    return (1 + lr*v + lphi*(u/r) + 
            lu*(-u*v/r + params['T']*np.cos(beta)/m) + 
            lv*(u**2/r - params['mu']/r**2 + params['T']*np.sin(beta)/m))