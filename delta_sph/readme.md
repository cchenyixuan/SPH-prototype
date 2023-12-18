$$
\left< f \right> \left( r_i \right) = \sum_j f_j W\left( r_i-r_j, h \right) V_j
$$

$$
\left\{
\begin{aligned}
\frac{D\rho}{Dt}&=-\rho\nabla\cdot u \\
p&=c_0^2\left( \rho - \rho_0 \right) \\
\rho \frac{Du}{Dt}&=-\nabla p + \rho f
\end{aligned}

\right.
$$

$$
\left\{
\begin{aligned}

\frac{D\rho_i}{Dt}&=-\rho_i \sum_j \left( u_j - u_i \right) \cdot \nabla_i W\left( r_j, h \right) V_j + \delta h c_0\sum_j \psi_{ij}\cdot \nabla_i W\left( r_j, h \right) V_j \\
p_i&=c_0^2\left( \rho_i - \rho_0 \right) \\
\rho_i \frac{Du_i}{Dt}&=-\sum_j \left( p_j + p_i \right)\nabla_i W\left( r_j, h \right) V_j + \rho_i f_i + \alpha h c_0 \rho_0 \sum_j \pi_{ij} \nabla_i W\left( r_j, h \right) V_j

\end{aligned}

\right.
$$

$$
\psi_{ij} = 2\left( \rho_j - \rho_i \right)\frac{r_j-r_i}{|rij|^2}-\left[ \left< \nabla \rho \right>_i^L + \left< \nabla \rho \right>_j^L \right]
$$

$$
\pi_{ij} = \frac{\left( u_j - u_i \right)\cdot \left(r_j-r_i\right)}{|r_{ij}|^2}
$$

$$
\left< \nabla \rho \right>_a^L = \sum_b \left( \rho_b - \rho_a \right) L_a\nabla_a W\left( r_j, h \right) V_j
$$

$$
L_a = \left[ \sum_b \left(r_b-r_a\right)\otimes \nabla_a W\left( r_b, h \right) V_b \right]^{-1}
$$

$$
u=\left[
\begin{aligned}
&u_1 \\
&u_2 \\
&\space \vdots \\
&u_m \\

\end{aligned}
\right],
v=\left[
\begin{aligned}
&v_1 \\
&v_2 \\
&\space \vdots \\
&v_n \\

\end{aligned}
\right],
u\otimes v=\left[
\begin{aligned}
&u_1v_1       &\quad  &u_1v_2        &\quad\cdots  &\quad  &u_1v_n           \\
&u_2v_1       &\quad  &u_2v_2        &\quad\cdots  &\quad  &u_2v_n           \\
&\quad\vdots  &\quad  &\quad\vdots   &\quad\ddots  &\quad  &\quad\vdots\quad \\
&u_mv_1       &\quad  &u_mv_2        &\quad\cdots  &\quad  &u_mv_n           \\

\end{aligned}
\right]
$$

```glsl

# version 460 core
```
**Momentum equation defined in JOSEPHINE and the delta-SPH model**

$$
        \frac{du_{i}}{dt}=-\frac{1}{\rho_{i}}\sum_{j}\left(p_{j}+p_{i}\right)
        \nabla_{i}W_{ij}V_{j}+\mathbf{g}_{i}+\alpha hc_{0}\rho_{0}\sum_{j}
        \pi_{ij}\nabla_{i}W_{ij}V_{j}
$$
    where

$$
        \pi_{ij}=\frac{\mathbf{u}_{ij}\cdot\mathbf{r}_{ij}}
        {|\mathbf{r}_{ij}|^{2}}
$$


**Continuity equation with dissipative terms**

$$
    \frac{d\rho_a}{dt} = \sum_b \rho_a \frac{m_b}{\rho_b}
    \left( \boldsymbol{v}_{ab}\cdot \nabla_a W_{ab} + \delta \eta_{ab}
    \cdot \nabla_{a} W_{ab} (h_{ab}\frac{c_{ab}}{\rho_a}(\rho_b -
    \rho_a)) \right)
$$
