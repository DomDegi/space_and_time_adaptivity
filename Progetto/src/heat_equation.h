/**
 * @file heat_equation.h
 * @brief Header for the main HeatEquation solver class and statistics structure.
 */

#ifndef HEAT_EQUATION_H
#define HEAT_EQUATION_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include <string>
#include <limits>
#include <filesystem>

// Include the source term definition
#include "right_hand_side.h"

namespace Progetto
{
  using namespace dealii;

  /**
   * @class HeatEquation
   * @brief Main class for solving the time-dependent Heat Equation.
   *
   * This class manages the Finite Element Method (FEM) simulation, including:
   * - Grid generation or loading.
   * - System assembly (Mass and Stiffness matrices).
   * - Time stepping (Theta-scheme: Euler or Crank-Nicolson).
   * - Adaptivity (Mesh refinement and Time step sizing).
   * - Output generation (VTK files and CSV logs).
   *
   * @tparam dim Spatial dimension (e.g., 2).
   */
  template <int dim>
  class HeatEquation
  {
  public:
    /**
     * @struct RunStats
     * @brief Container for collecting performance and simulation statistics.
     * * Used to compare different configurations (Fixed vs Adaptive).
     */
    struct RunStats
    {
      double cpu_seconds_total = 0.0;     ///< Total execution time

      // Solver statistics
      unsigned int n_linear_solves = 0;
      unsigned long long cg_iterations_sum = 0;
      unsigned int cg_iterations_min = std::numeric_limits<unsigned int>::max();
      unsigned int cg_iterations_max = 0;

      // Time stepping statistics
      unsigned int n_time_steps_total = 0;
      unsigned int n_time_steps_accepted = 0;
      unsigned int n_time_steps_rejected = 0;

      double dt_min = std::numeric_limits<double>::max();
      double dt_max = 0.0;
      double dt_sum = 0.0;

      // Mesh statistics
      unsigned long long dof_sum = 0;
      unsigned int dof_min = std::numeric_limits<unsigned int>::max();
      unsigned int dof_max = 0;
      unsigned int dof_samples = 0;
      unsigned int cells_min = std::numeric_limits<unsigned int>::max();
      unsigned int cells_max = 0;

      // Methods to update stats
      void reset();
      void sample_dofs_and_cells(const unsigned int ndofs, const unsigned int ncells);
      void register_cg_iterations(const unsigned int it);
      void register_time_step_attempt(const double dt, const bool accepted);

      // Getters for averages
      double dt_mean() const;
      double dof_mean() const;
    };

    /**
     * @brief Constructor initializing all simulation parameters.
     * * See the implementation file for detailed parameter descriptions.
     */
    HeatEquation(const std::string &run_name_in,
                 const std::string &output_dir_in,
                 bool space_adaptivity,
                 bool time_adaptivity,
                 bool step_doubling,
                 bool mesh,
                 int  cells_per_direction,
                 unsigned int rhs_N_in,
                 double rhs_sigma_in,
                 double rhs_a_in,
                 double rhs_x0_x_in,
                 double rhs_x0_y_in,
                 double material_diffusion_in,
                 double material_mass_in,
                 double theta_in,
                 double time_step_tolerance_in,
                 double time_step_min_in,
                 double end_time_in,
                 double initial_dt_in,
                 unsigned int initial_global_refinement_in,
                 unsigned int n_adaptive_pre_refinement_steps_in,
                 unsigned int refine_every_n_steps_in,
                 bool write_vtk_in);

    /**
     * @brief Main driver function.
     * * Executes the simulation loop: setup -> time loop -> solve -> output.
     */
    void run();

    // --- Getters for analysis ---
    const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }
    const Vector<double>  &get_solution()   const { return solution; }
    const RunStats        &get_stats()      const { return stats; }

    /**
     * @brief Computes L2 error against a provided analytical or reference function.
     */
    double compute_L2_error_against(const Function<dim> &reference_function) const;

  private:
    /**
     * @brief Sets up the DoFHandler, matrices, and vectors.
     * Resizes system matrix, mass matrix, and laplace matrix.
     */
    void setup_system();

    /**
     * @brief Solves the linear system M*x = RHS using CG solver.
     */
    void solve_time_step();

    /**
     * @brief Performs a single time step integration.
     * * Assembles the RHS based on the theta-scheme and solves for u_out.
     * * @param u_old Solution at previous time step.
     * @param dt    Time step size.
     * @param t_new Target time.
     * @param u_out Output vector to store the new solution.
     */
    void do_time_step(const Vector<double> &u_old,
                      const double          dt,
                      const double          t_new,
                      Vector<double> &      u_out);

    /**
     * @brief Outputs the current solution to a VTK file.
     */
    void output_results() const;

    /**
     * @brief Refines the mesh based on Kelly error estimator.
     * * Uses solution transfer to map the old solution to the new mesh.
     */
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    // --- Logging Helpers ---
    void log_mesh_event(const unsigned int marked_refine,
                        const unsigned int marked_coarsen) const;

    void log_time_event(const double t_now,
                        const double dt_now,
                        const int accepted,
                        const double err_est,
                        const double new_dt) const;

    // --- DEAL.II Objects ---
    Triangulation<dim> triangulation;
    const FE_Q<dim>    fe;
    DoFHandler<dim>    dof_handler;
    AffineConstraints<double> constraints;

    // Linear Algebra
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;     ///< Mass matrix (M)
    SparseMatrix<double> laplace_matrix;  ///< Stiffness/Laplace matrix (A)
    SparseMatrix<double> system_matrix;   ///< Matrix to solve: M + theta*dt*A

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    // --- Simulation State ---
    double       time = 0.0;
    double       time_step;
    unsigned int timestep_number = 0;

    // --- Configuration Flags ---
    bool use_space_adaptivity;
    bool use_time_adaptivity;
    bool use_step_doubling;

    // --- Time Stepping Parameters ---
    double time_step_tolerance;
    double time_step_min;
    double time_step_max = 1e-1;
    double time_step_safety = 0.9;
    const double theta; ///< 0=Explicit, 0.5=Crank-Nicolson, 1.0=Implicit

    // --- Material Parameters ---
    double material_diffusion; ///< Diffusion coefficient (k)
    double material_mass;      ///< Reaction/Mass coefficient (c)

    // --- Mesh & RHS Parameters ---
    bool use_mesh;
    int  cells_per_direction;
    unsigned int rhs_N;
    double rhs_sigma;
    double rhs_a;
    Point<dim> rhs_x0;

    double end_time;

    unsigned int initial_global_refinement;
    unsigned int n_adaptive_pre_refinement_steps;
    unsigned int refine_every_n_steps;

    bool write_vtk;

    std::string run_name;
    std::string output_dir;

    RunStats stats;
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
  };
}

#endif // HEAT_EQUATION_H