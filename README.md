**TSDynamics**: A Python package for defining, simulating, and analyzing dynamical systems and time series with a user-friendly interface.

### Description:

TSDynamics is designed to simplify the numerical study of dynamical systems and time series analysis, offering a seamless interface for researchers and practitioners. The package provides tools for defining, simulating, and exploring various types of systems and their behavior over time, including:

- **Continuous Dynamical Systems**: Easily define systems by providing the right-hand side (RHS) of differential equations.
- **Time-Delay Systems**: Include delays in the system dynamics effortlessly by specifying delayed terms in the RHS.
- **Discrete Maps**: Define iterative systems and discrete-time dynamics with minimal setup.

### Key Features (Current):

1. **Straightforward System Definition**:
   - Users can specify the RHS of their systems as Python functions or callable objects.
   - Flexible support for both continuous and discrete systems.

2. **Support for Time Delays**:
   - Define delay differential equations (DDEs) without cumbersome setup.
   - Seamless integration of delayed variables in the RHS definition.

3. **Discrete Maps**:
   - Analyze iterative systems by defining discrete updates.
   - Ideal for studying chaotic maps, fixed points, and bifurcations.

4. **Simulation Tools**:
   - Numerical solvers optimized for dynamical systems.
   - Support for flexible time steps and adaptive methods.

### Planned Features (Inspired by TiSEAN):

1. Advanced tools for analyzing time series:
   - Lyapunov exponents.
   - Recurrence plots.
   - Surrogate data testing.
   - Dimension estimation and embedding.

2. Phase-space visualization tools for qualitative analysis.

3. Extensions for stochastic dynamics and noise-driven systems.

4. Robust handling of high-dimensional systems and parallel simulations.

### Use Cases:

- Exploring the behavior of dynamical systems (e.g., chaos, bifurcations).
- Simulating delay differential equations.
- Investigating discrete maps and iterative systems.
- Time series analysis for scientific, engineering, and data-driven applications.

TSDynamics aims to be a versatile and extensible toolkit for numerical dynamics, bridging the gap between ease of use and advanced functionality.