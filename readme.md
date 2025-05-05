```markdown
# Lunar Landing Trajectory Optimization

Optimal control solution for a lunar landing trajectory using indirect methods and shooting technique.

## Problem Formulation

### System Dynamics
The lunar landing vehicle is modeled with the following state equations:


| **Radial Position** | `dr/dt = v` |
| **Angular Position** | `dφ/dt = u/r` |
| **Tangential Velocity** | `du/dt = -uv/r + (T/m)cosβ` |
| **Radial Velocity** | `dv/dt = u²/r - μ/r² + (T/m)sinβ` |


Where:
- **r**: Radial distance from moon center [m]
- **φ**: Angular position [rad]
- **u**: Tangential velocity [m/s]
- **v**: Radial velocity [m/s]
- **T**: Thrust magnitude [N]
- **m**: Vehicle mass [kg]
- **β**: Thrust angle [rad]
- **μ**: Lunar gravitational parameter [m³/s²]

### Costate Equations
The Hamiltonian is given by:

```math
H = 1 + λ_r v + λ_φ (u/r) + λ_u (-uv/r + (T/m)cosβ) + λ_v (u²/r - μ/r² + (T/m)sinβ)
```

The costate equations are:

```
math
<div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
  <table>
    <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;"><b>Radial Position</b></td>
        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><code>dr/dt = v</code></td></tr>
    <tr><td style="padding: 8px;"><b>Angular Position</b></td>
        <td style="padding: 8px;"><code>dφ/dt = u/r</code></td></tr>
  </table>
</div>
```

### Optimal Control Law
The optimal thrust angle is determined by:

```math
β^* = atan2(λ_v, λ_u)
```

## Boundary Conditions

**Initial Conditions (t=0):**
```
r(0) = R_moon + 15 km
φ(0) = π/2 rad
u(0) = 1691.92 m/s
v(0) = 0 m/s
```

**Terminal Conditions (t=tf):**
```
r(tf) = R_moon
u(tf) = 0 m/s
v(tf) = -5 m/s (soft landing)
```

## Solution Method

The problem is solved using:
1. **Indirect Shooting Method**
2. **Boundary Value Problem** formulation
3. **Numerical Optimization** (BFGS/L-BFGS-B) to find initial costates

### Numerical Approach
```python
def shooting_obj(x, x0, xf, params):
    # x = [λ_r, λ_u, λ_v, log(tf)]
    lambda0 = [x[0]*1e-6, 0, x[1]*1e-3, x[2]*1e-3]
    tf = exp(x[3])
    
    # Solve BVP
    sol = solve_ivp(dynamics, [0,1], [x0; lambda0], 
                   args=(tf,params), method='LSODA')
    
    # Calculate errors
    pos_err = (rf - xf[0])/R_moon
    vel_err = [uf/xf[1], (vf-xf[2])/abs(xf[2])]
    
    return weighted_errors + Hamiltonian_constraint
```

## Code Structure

```
lunar_landing/
├── README.md               # This document
├── main.py                 # Main optimization routine
├── dynamics.py             # System dynamics and costate equations
├── plotting.py             # Visualization functions
└── requirements.txt        # Python dependencies
```

## Results Example

```
====== FINAL RESULTS ======
Flight Time: 990.93 s
Final Altitude: -0.00 km (Target: 0.0 km)
Final Velocities: u=0.00 m/s (Target 0), v=-5.00 m/s (Target -5.0)
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lunar-landing.git
cd lunar-landing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the optimization:
```bash
python main.py
```

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## References

1. Bryson, A. E., & Ho, Y. (1975). *Applied Optimal Control*.
2. Betts, J. T. (1998). *Survey of Numerical Methods for Trajectory Optimization*.
```

### Key Features:
1. Proper mathematical formatting using GitHub's LaTeX-like syntax
2. Clear problem formulation with equations
3. Complete documentation of the solution method
4. Code structure overview
5. Example results section
6. Getting started guide

### For Better Math Rendering:
GitHub's markdown supports limited LaTeX. For better rendering:
1. Use online LaTeX editors for complex equations
2. Render equations as images if needed
3. Consider hosting documentation on ReadTheDocs with full LaTeX support

### How Equations Appear on GitHub:
- Inline equations: `$H = ...$` 
- Display equations:
  ```math
  \frac{dr}{dt} = v
  ```
- Matrix equations:
  ```
  [ λ_r ]   [ ∂H/∂r ]
  [ λ_φ ] = [ ∂H/∂φ ]
  ```

