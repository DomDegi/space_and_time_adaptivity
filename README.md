# Project 5: Space and Time Adaptivity

This project implements an adaptive space–time finite element solver for the heat equation with localized impulsive forcing. The solver utilizes **deal.II** to handle **adaptive mesh refinement (AMR)** in space and **adaptive time-stepping** to efficiently resolve dynamic physical phenomena.

## Problem Formulation

The solver addresses the heat equation with homogeneous Neumann boundary conditions:

$$
\begin{cases}
\frac{\partial u}{\partial t} - \nabla \cdot (\mu \nabla u) = f & \text{in } \Omega \times (0,T), \\
\mu \nabla u \cdot \mathbf{n} = 0 & \text{on } \partial\Omega \times (0,T), \\
u(\mathbf{x},0) = 0 & \text{in } \Omega.
\end{cases}
$$

**Default Configuration:**
* **Diffusion:** $\mu = 1$.
* **Forcing Term ($f$):** A sequence of impulses, localized in both space and time, defined as $f(x,t) = g(t)h(x)$, where:
    * $g(t) = \frac{\exp(-a \cos(2N\pi t))}{\exp(a)}$ (Temporal pulsation).
    * $h(x) = \exp\left(-\frac{|x-x_0|^2}{\sigma^2}\right)$ (Spatial Gaussian).

---

## Build & Execution Instructions

This project is designed to run inside the provided Apptainer container `amsc_mk_2025.sif`.

### 1. Environment Setup
Before compiling, ensure you are inside the container and have loaded the necessary modules:

1.  **Enter the container:**
    ```bash
    apptainer shell amsc_mk_2025.sif
    ```
2.  **Load the modules:**
    ```bash
    module load gcc-glibc dealii
    ```

### 2. Compile and Run

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone [https://github.com/DomDegi/space_and_time_adaptivity.git](https://github.com/DomDegi/space_and_time_adaptivity)
    cd space_and_time_adaptivity/
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure with CMake:**
    ```bash
    cmake ..
    ```

4.  **Compile:**
    ```bash
    make
    ```

5.  **Run the program:**
    ```bash
    ./heat-equation
    ```
    Or specify a custom parameter file:
    ```bash
    ./heat-equation my_parameters.prm
    ```
    *(Note: The `solutions` directory will be created automatically by the program).*

---

## Usage & Configuration

### Parameter File

The program uses a **parameter file** (`parameters.prm`) to configure all simulation settings. This approach is suitable for large-scale computing where jobs are submitted to schedulers, and allows easy modification without recompilation.

**On first run**, if no parameter file exists, the program will automatically create a default `parameters.prm` with all available options and documentation. Simply edit this file and run the program again.

**Parameter file sections:**

#### 1. Mesh Configuration
```prm
subsection Mesh
  set generate_mesh = true          # Generate mesh internally or load from file
  set cells_per_direction = 10      # Number of cells per direction (if generated)
end
```

#### 2. Physical Parameters
Set the parameters for the heat source and domain:
```prm
subsection Physical Parameters
  set final_time = 0.5                    # Final simulation time T
  set source_width_sigma = 0.5            # Gaussian width σ (must be positive)
  set source_center_x = 0.5               # Source center x-coordinate
  set source_center_y = 0.5               # Source center y-coordinate
  set source_frequency_N = 5              # Oscillation frequency N
  set oscillation_magnitude_A = 5.0       # Amplitude parameter A
end
```

#### 3. Material Properties
Configure the physical material coefficients:
```prm
subsection Material Properties
  set density = 1.0                       # Density ρ [kg/m³] (must be positive)
  set specific_heat = 1.0                 # Specific heat c_p [J/(kg·K)] (must be positive)
  set thermal_conductivity = 1.0          # Thermal conductivity k [W/(m·K)] (must be positive)
  set source_intensity = 1.0              # Source intensity Q [W/m³]
end
```

This generalizes the equation to the full heat transfer form:

$$
\rho c_p \frac{\partial u}{\partial t} - \nabla \cdot (k \nabla u) = Q \cdot g(t)h(\mathbf{x})
$$

⚠️ **Important:** All material parameters (ρ, c_p, k) must be **strictly positive**.
> Setting these values to zero or negative numbers will violate the **Symmetric Positive Definite (SPD)** property required by the Conjugate Gradient (CG) solver, causing immediate runtime errors (e.g., `NaN` residuals).

#### 4. Solver Settings
```prm
subsection Solver Settings
  set theta = 0.5                         # Time integration: 0=Explicit, 0.5=Crank-Nicolson, 1=Implicit
  set time_step_tolerance = 1e-6          # Error threshold for adaptivity
  set minimum_time_step = 1e-4            # Minimum allowed time step
  set use_step_doubling = true            # Use step-doubling (true) or heuristic (false)
end
```

#### 5. Simulation Control
```prm
subsection Simulation Control
  set run_mode = 0                        # 0=Full comparison, 1-4=specific configurations
  set run_reference = true                # Run high-resolution reference for L2 errors
  set initial_time_step = 0.002           # Initial time step size
  set base_refinement = 2                 # Base global refinement level
  set pre_refinement_steps = 4            # Number of pre-refinement steps
  set refine_every_n_steps = 5            # Refine mesh every N steps
  set write_vtk = true                    # Write VTK output files
end
```

**Simulation Modes:**

| Mode | Description |
| :--- | :--- |
| **0** | **Full Comparison (Benchmark)**<br>Runs a High-Res Reference solution, then runs ALL 4 configurations and saves a summary CSV. |
| **1** | **Fixed Space + Fixed Time**<br>Standard FEM on a uniform grid with constant Δt. |
| **2** | **Adaptive Space + Fixed Time**<br>Uses AMR (Kelly Error Estimator) but keeps Δt constant. |
| **3** | **Fixed Space + Adaptive Time**<br>Uniform grid, but varies Δt based on temporal error. |
| **4** | **Adaptive Space + Adaptive Time**<br>Full adaptivity in both space and time. |

### Legacy Interactive Mode (Deprecated)

Previous versions used an interactive CLI. This has been replaced with the parameter file approach for better reproducibility and scriptability. The interactive helper functions remain in `utilities.cc` for potential debugging use.

---

## Output

Results are saved in the `build/solutions/` directory:
* **VTK Files:** `solution-00001.vtk`, etc. (Open with Paraview or VisIt).
* **CSV Logs:** `time_log.csv` (step sizes) and `mesh_log.csv` (DoF counts).
* **Summary:** `summary_comparison.csv` (created in Mode 0 for performance analysis).

---

## Documentation

This project uses **Doxygen** to generate code documentation and dependency graphs.

### Generation
Since Doxygen is already available in the container, simply run the following command in the project root:
```bash
doxygen Doxyfile
```
You can view the documentation by opening the index.html file located in docs/html/ inside your browser.
