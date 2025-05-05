import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dynamics import dynamics, compute_hamiltonian
from plotting import plot_solution

class OptimizationTracker:
    def __init__(self, x0, xf, params):
        self.x0 = x0
        self.xf = xf
        self.params = params
        self.iteration = 0
        
    def __call__(self, xk):
        self.iteration += 1
        if self.iteration % 10 == 0:  # Only print every 10 iterations
            cost = shooting_obj(xk, self.x0, self.xf, self.params)
            print(f"\nIteration {self.iteration}:")
            print(f"Current cost: {cost:.4e}")
            print(f"Parameters: λr={xk[0]:.4f}, λu={xk[1]:.4f}, λv={xk[2]:.4f}, log(tf)={xk[3]:.4f}")
        return False

def lunar_landing_optimization():
    """Main optimization routine with improved convergence"""
    params = {
        'T': 440, 'M0': 300, 'mdot': 0.15,
        'mu': 4.9026e12, 'RM': 1.7382e6
    }
    
    x0 = np.array([params['RM'] + 15e3, np.pi/2, 1691.92, 0])
    xf = np.array([params['RM'], 0, -5])
    
    # Better initial guess with bounds
    init_guess = np.array([-2.5, -1e-2, 1e-2, np.log(1000)])
    bounds = Bounds(
        [-10, -0.1, -0.1, np.log(300)],  # Lower bounds
        [10, 0.1, 0.1, np.log(1500)]     # Upper bounds
    )
    
    tracker = OptimizationTracker(x0, xf, params)
    
    # Try multiple optimization methods
    methods = ['BFGS', 'L-BFGS-B', 'TNC']
    best_solution = None
    best_cost = np.inf
    
    for method in methods:
        print(f"\n=== Trying {method} optimization ===")
        res = minimize(
            shooting_obj, init_guess, 
            args=(x0, xf, params),
            method=method,
            bounds=bounds if method in ['L-BFGS-B', 'TNC'] else None,
            callback=tracker,
            options={
                'maxiter': 1000,
                'gtol': 1e-10,
                'xtol': 1e-10,
                'disp': True
            }
        )
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_solution = res.x
    
    if best_solution is not None:
        print("\n=== Best Solution Found ===")
        plot_solution(best_solution, x0, xf, params)
        plt.show(block=True)
    else:
        print("Optimization failed to converge")

def shooting_obj(x, x0, xf, params):
    """Enhanced objective function with better normalization"""
    lambda0 = np.array([x[0]*1e-6, 0, x[1]*1e-3, x[2]*1e-3])
    tf = np.exp(x[3])
    
    try:
        sol = solve_ivp(
            lambda t, y: dynamics(t, y, tf, params),
            [0, 1], np.concatenate([x0, lambda0]),
            method='LSODA',
            rtol=1e-10,
            atol=1e-12
        )
        Y = sol.y.T[-1]
    except:
        return 1e15  # Higher penalty for failures
    
    # Normalized errors with better scaling
    pos_err = (Y[0] - xf[0]) / params['RM']  # Normalized by lunar radius
    u_err = Y[2] / max(abs(xf[1]), 0.1)      # Normalized by target velocity
    v_err = (Y[3] - xf[2]) / max(abs(xf[2]), 0.1)
    
    # Hamiltonian should be zero for optimal solution
    H = compute_hamiltonian(Y, tf, params)
    
    # Weighted cost function - prioritize position and vertical velocity
    weights = np.array([1e12, 1e6, 1e8, 100])  # pos, u, v, H
    errors = np.array([pos_err, u_err, v_err, H])
    
    return np.dot(weights, errors**2)

if __name__ == "__main__":
    lunar_landing_optimization()