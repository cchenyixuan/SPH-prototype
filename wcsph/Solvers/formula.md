Equation:

$$
\rho \frac{Du}{Dt} = -\nabla p + \mu \Delta u + f
$$

$$
\left\{
\begin{aligned}\rho \frac{Du}{Dt} &= -\nabla p + \mu_{material} \Delta u + (\mu_{material} + \mu_{trans}) \nabla \times \omega + f \\ \theta \rho \frac{D\omega}{Dt} &= \mu_{material} \Delta \omega + (\mu_{material} + \mu_{trans}) (\nabla \times u - 2\omega) + \tau 
\end{aligned}
\right.
$$

where $\rho$ is the fluid density, $p$ is the pressure, $u$ is the fluid velocity, $\omega$ is the angular velocity, $f, \tau$ are the external force and torque, $\mu_{material}$ is the viscosity factor of the material, $\mu_{trans}$ is the transfer factor of linear momentum to angular momentum and $\theta$ is the microinertia coefficient which we set to $\theta=2$ in all our calculations.

Weakly-Compressible Navier-Stokes Equation.

Density:

$$
\rho = \sum_j m_j W_{ij}
$$

Pressure:

$$
p = \max\left\{\frac{{C_0}^2\rho_0}{\gamma}\left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_0, 0\right\}
$$

Pressure Gradient:

$$
\nabla p_i = \rho_i \sum_j m_j \left( \frac{p_i}{{\rho_i}^2} + \frac{p_j}{{\rho_j}^2} \right) \nabla W_{ij}
$$

$$
\nabla p_i = \sum_j \frac{m_j}{\rho_j} (p_j - p_i) \nabla W_{ij}
$$

Velocity Laplacian:

$$
\Delta u_i = \sum_j \frac{m_j}{\rho_j} (u_j - u_i) \frac{2\|\nabla W_{ij}\|}{\|r_{ij}\|}
$$

$$
\Delta u_i = 2(d+2)\sum_j \frac{m_j}{\rho_j} \frac{u_{ij}\cdot r_{ij}}{{\|r_{ij}\|}^2 + 0.01h^2} \nabla W_{ij}
$$

Velocity Curl:

$$
\nabla \times u_i = -\rho_i \sum_j m_j \left( \frac{u_i}{{\rho_i}^2} + \frac{u_j}{{\rho_j}^2} \right) \times \nabla W_{ij}
$$

$$
\nabla \times u_i = \sum_j \frac{m_j}{\rho_j} (u_i - u_j) \nabla W_{ij}
$$

$$
\nabla \times u_i = \frac{1}{\rho_i} \sum_j m_j (u_i - u_j) \nabla W_{ij}
$$

Angular Velocity Curl:

$$
\nabla \times \omega_i = -\rho_i \sum_j m_j \left( \frac{\omega_i}{{\rho_i}^2} + \frac{\omega_j}{{\rho_j}^2} \right) \times \nabla W_{ij}
$$

$$
\nabla \times \omega_i = \sum_j \frac{m_j}{\rho_j} (\omega_i - \omega_j) \nabla W_{ij}
$$

$$
\nabla \times \omega_i = \frac{1}{\rho_i} \sum_j m_j (\omega_i - \omega_j) \nabla W_{ij}
$$

XSPH Form:

$$
u_i = u_i + \alpha \sum_j \frac{m_j}{\rho_j}(u_j - u_i)W_{ij}
$$

$$
\omega_i = \omega_i + \beta \sum_j \frac{m_j}{\rho_j}(\omega_j - \omega_i)W_{ij}
$$

Parameters:

$$
\alpha = 0.002, \beta = 0.125,\mu_{trans}<0.4
$$

Boundary Particles Mass (Mass only consider the neighborhood boundary particles):

$$
m_i = \frac{\rho_0}{\sum_j W_{ij}}
$$

Boundary Sampling Form:

$$


$$




Idea: calculate the volume sum of boundary when a fluid gets closed

the repause force only activate when some sum function arrive certain value, else the boundary shall not influence the fluid domain
