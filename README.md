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
    *(Note: The `solutions` directory will be created automatically by the program).*

---

## Usage & Configuration

Upon running `./heat-equation`, the application will launch an interactive command-line interface (CLI). You will be prompted to configure the simulation in the following order:

### 1. Mesh Generation
* **Generate internal grid?**
    * `y` (Yes): Generates a structured HyperCube grid internally. You will be asked for the number of **cells per direction** (e.g., 20).
    * `n` (No): Loads an external mesh file (expected at `../mesh/mesh-input.msh`).

### 2. Physical Parameters
Set the parameters for the heat source and domain (corresponding to the equations above):
* **Final Time ($T$):** The duration of the simulation (default: `0.5`).
* **Source Width ($\sigma$):** The Gaussian width of the heat source (default: `0.5`).
* **Source Center ($x_0, y_0$):** Coordinates of the source center (default: `0.5, 0.5`).
* **Frequency ($N$):** The oscillation frequency of the source term (default: `5`).
* **Magnitude ($A$):** The amplitude of the source oscillation (default: `5.0`).

### 3. Material Properties (Coefficients)
The program asks: *"Do you want to customize physical material coefficients?"*
* **No:** Uses default unitary values ($\mu=1$, equivalent to $\rho=1, c_p=1, k=1$).
* **Yes:** You can specify the actual physical properties. This generalizes the equation to the full heat transfer form:

$$
\rho c_p \frac{\partial u}{\partial t} - \nabla \cdot (k \nabla u) = Q \cdot g(t)h(\mathbf{x})
$$

Where the user-defined parameters are:
* **Density ($\rho$):** Mass density of the material [kg/m³].
* **Specific Heat ($c_p$):** Heat capacity [J/(kg·K)].
* **Thermal Conductivity ($k$):** Rate of heat transfer [W/(m·K)].
* **Source Intensity ($Q$):** Maximum volumetric power density [W/m³].

⚠️ **Important:** All material parameters ($\rho, c_p, k$) must be **strictly positive**.
> Setting these values to zero or negative numbers will violate the **Symmetric Positive Definite (SPD)** property required by the Conjugate Gradient (CG) solver, causing immediate runtime errors (e.g., `NaN` residuals).

### 4. Solver & Time Stepping (Optional)
The program asks: *"Do you want to customize solver & time adaptivity settings?"*
* **No:** Uses defaults ($\theta=0.5$ Crank-Nicolson, Tol=10^{-6}).
* **Yes:** You can specify:
    * **Theta ($\theta$):** Time integration scheme.
        * `0.0`: Forward Euler (Explicit).
        * `0.5`: Crank-Nicolson (Implicit, 2nd order, default).
        * `1.0`: Backward Euler (Implicit, 1st order, stable).
    * **Tolerance:** The error threshold for time-step adaptivity (default `1e-6`).
    * **Min Time Step ($dt_{min}$):** The hard limit for the smallest allowed time step (default `1e-4`).

### 5. Adaptivity Strategy
* **Time Step Doubling:** You can choose between:
    * `y`: **Step Doubling** (Runs two half-steps and compares with one full step to estimate error). More accurate but computationally heavier.
    * `n`: **Heuristic** (Estimates error based on the change in the Right-Hand Side). Faster but less rigorous.

### 6. Simulation Modes
Finally, select which configuration(s) to run:

| Mode | Description |
| :--- | :--- |
| **0** | **Full Comparison (Benchmark)**<br>Runs a High-Res Reference solution, then runs ALL 4 configurations below and saves a summary CSV comparing L2 errors and CPU time. |
| **1** | **Fixed Space + Fixed Time**<br>Standard FEM on a uniform grid with constant $dt$. |
| **2** | **Adaptive Space + Fixed Time**<br>Uses AMR (Kelly Error Estimator) but keeps $dt$ constant. |
| **3** | **Fixed Space + Adaptive Time**<br>Uniform grid, but varies $dt$ based on temporal error. |
| **4** | **Adaptive Space + Adaptive Time**<br>Full adaptivity in both space and time. |

*Note: If you select modes 1-4, the program will ask if you want to run the **Reference Solver** first. This is required if you want to calculate and see the L2 Error in the output.*

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
