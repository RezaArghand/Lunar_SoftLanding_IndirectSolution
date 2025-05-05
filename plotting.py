import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dynamics import dynamics, compute_hamiltonian

def plot_solution(xopt, x0, xf, params):
    """Main plotting function with proper parameter handling"""
    plt.ioff()  # Disable interactive mode to control figure display
    # Unpack optimized parameters
    lambda0 = np.array([xopt[0]*1e-6, 0, xopt[1]*1e-3, xopt[2]*1e-3])
    tf = np.exp(xopt[3])
    
    # High-resolution simulation
    sol = solve_ivp(
        lambda t, y: dynamics(t, y, tf, params),
        [0, 1], np.concatenate([x0, lambda0]),
        t_eval=np.linspace(0, 1, 1000),
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    t = sol.t * tf
    Y = sol.y.T
    
    # Create plots
    plt.close('all')
    plot_trajectory(t, Y, xf, params, tf)
    plot_costates(t, Y, params)  # Pass params here
    plot_control_and_energy(t, Y, params)
    plt.show(block=True)  # Ensures plots remain visible
    
    print_results(Y[-1], tf, xf, params)

def plot_trajectory(t, Y, xf, params, tf):
    """Plot altitude and velocity states"""
    plt.figure(figsize=(10, 8))
    altitude = (Y[:, 0] - params['RM'])/1000
    
    plt.subplot(3, 1, 1)
    plt.plot(t, altitude, 'b-', linewidth=2)
    plt.ylabel('Altitude (km)')
    plt.title(f'Optimal Lunar Landing (tf = {tf:.2f} s)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, Y[:, 2], 'r-', linewidth=2)
    plt.axhline(0, color='k', linestyle='--')
    plt.ylabel('Horizontal Velocity (m/s)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, Y[:, 3], 'g-', linewidth=2)
    plt.axhline(xf[2], color='k', linestyle='--')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()

def plot_costates(t, Y, params):  # Added params parameter
    """Plot costate variables evolution"""
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(t, np.abs(Y[:, 4:8]), linewidth=1.5)
    plt.legend(['λr', 'λφ', 'λu', 'λv'])
    plt.ylabel('Costate Magnitude (log scale)')
    plt.title('Costate Variables')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    H = np.array([compute_hamiltonian(Y[i], t[i], params) 
                 for i in range(len(t))])
    plt.plot(t, H, 'm-', linewidth=2)
    plt.axhline(0, color='k', linestyle='--')
    plt.ylabel('Hamiltonian')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()

def plot_control_and_energy(t, Y, params):
    """Plot control and energy components"""
    plt.figure(figsize=(10, 8))
    
    # Control angle
    beta = np.arctan2(Y[:, 7], Y[:, 6])
    plt.subplot(2, 2, 1)
    plt.plot(t, np.rad2deg(beta), 'c-', linewidth=2)
    plt.ylabel('Thrust Angle (deg)')
    plt.grid(True)
    
    # Mass
    m = np.maximum(params['M0'] - params['mdot']*t, 10)
    plt.subplot(2, 2, 2)
    plt.plot(t, m, 'm-', linewidth=2)
    plt.ylabel('Mass (kg)')
    plt.grid(True)
    
    # Energy
    kinetic = 0.5*(Y[:, 2]**2 + Y[:, 3]**2)
    potential = params['mu']/Y[:, 0]
    plt.subplot(2, 2, (3, 4))
    plt.plot(t, kinetic, 'b-', label='Kinetic')
    plt.plot(t, potential, 'g-', label='Potential')
    plt.plot(t, kinetic+potential, 'r-', label='Total')
    plt.legend()
    plt.ylabel('Energy (J/kg)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()

def print_results(Y_final, tf, xf, params):
    """Print comprehensive results"""
    rf, uf, vf = Y_final[0], Y_final[2], Y_final[3]
    
    print("\n====== FINAL RESULTS ======")
    print(f"Flight Time: {tf:.2f} s")
    print(f"Final Altitude: {(rf-params['RM'])/1000:.2f} km (Target: {(xf[0]-params['RM'])/1000:.1f} km)")
    print(f"Final Velocities: u={uf:.2f} m/s (Target 0), v={vf:.2f} m/s (Target {xf[2]:.1f})")
    
    print("\n=== OPTIMIZATION PARAMETERS ===")
    print(f"Initial Costates: λr={Y_final[4]:.2e}, λu={Y_final[6]:.2e}, λv={Y_final[7]:.2e}")
    print(f"Final Hamiltonian: {compute_hamiltonian(Y_final, tf, params):.2e}")