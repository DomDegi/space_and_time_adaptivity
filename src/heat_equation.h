/**
 * @file heat_equation.h
 * @brief Header for the main HeatEquation solver class and statistics
 * structure.
 */

#ifndef HEAT_EQUATION_H
#define HEAT_EQUATION_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

// Trilinos Wrappers for parallel computation
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>

#include <filesystem>
#include <limits>
#include <string>

// Include the source term definition
#include "right_hand_side.h"

namespace Progetto
{
using namespace dealii;

/**
 * @struct HeatEquationParameters
 * @brief Parameter structure to configure HeatEquation solver.
 */
struct HeatEquationParameters
{
  // Run configuration
  std::string run_name;
  std::string output_dir;
  bool        use_space_adaptivity;
  bool        use_time_adaptivity;
  std::string time_adaptivity_method; // "step_doubling" or "heuristic"
  std::string time_step_controller;   // "integral" or "pi"
  bool        use_rannacher_smoothing;

  // Mesh configuration
  bool generate_mesh;
  int  cells_per_direction;

  // Physical parameters (source term)
  unsigned int source_frequency_N;
  double       source_width_sigma;
  double       source_magnitude_a;
  double       source_center_x;
  double       source_center_y;

  // Material properties
  double density;              // rho [kg/m^3]
  double specific_heat;        // c_p [J/(kg K)]
  double thermal_conductivity; // k [W/(m K)]
  double source_intensity;     // Q [W/m^3]

  // Solver settings
  double theta; // Time integration parameter
  double time_step_tolerance;
  double time_step_min;

  // Time stepping
  double end_time;
  double initial_time_step;

  // Refinement settings
  unsigned int initial_global_refinement;
  unsigned int n_adaptive_pre_refinement_steps;
  unsigned int refine_every_n_steps;

  // Output settings
  bool   write_vtk;
  bool   output_at_each_timestep;
  double output_time_interval;
};

/**
 * @class HeatEquation
 * @brief Main class for solving the time-dependent Heat Equation.
 *
 * Solves: rho * cp * du/dt - div(k * grad(u)) = Q
 */
template <int dim> class HeatEquation
{
public:
  /**
   * @struct RunStats
   * @brief Container for collecting performance and simulation statistics.
   */
  struct RunStats
  {
    double cpu_seconds_total = 0.0; ///< Total execution time

    unsigned int       n_linear_solves = 0;
    unsigned long long cg_iterations_sum = 0;
    unsigned int cg_iterations_min = std::numeric_limits<unsigned int>::max();
    unsigned int cg_iterations_max = 0;

    unsigned int n_time_steps_total = 0;
    unsigned int n_time_steps_accepted = 0;
    unsigned int n_time_steps_rejected = 0;

    double dt_min = std::numeric_limits<double>::max();
    double dt_max = 0.0;
    double dt_sum = 0.0;

    unsigned long long dof_sum = 0;
    unsigned int       dof_min = std::numeric_limits<unsigned int>::max();
    unsigned int       dof_max = 0;
    unsigned int       dof_samples = 0;
    unsigned int       cells_min = std::numeric_limits<unsigned int>::max();
    unsigned int       cells_max = 0;

    void   reset();
    void   sample_dofs_and_cells(const unsigned int ndofs,
                                 const unsigned int ncells);
    void   register_cg_iterations(const unsigned int it);
    void   register_time_step_attempt(const double dt, const bool accepted);
    double dt_mean() const;
    double dof_mean() const;
  };

  /**
   * @brief Constructor initializing all simulation parameters.
   * @param params Configuration parameters
   * @param comm MPI communicator (default: MPI_COMM_WORLD)
   */
  HeatEquation(const HeatEquationParameters &params,
               MPI_Comm                      comm = MPI_COMM_WORLD);

  /**
   * @brief Main driver function.
   *
   * Manages the overall simulation flow:
   * - Generates or loads mesh
   * - Sets up the linear system
   * - Time loop execution
   * - Adaptive mesh refinement (if enabled)
   * - Output generation
   */
  void run();

  /**
   * @brief Returns the DoFHandler (for reference solution verification).
   * @return const reference to the DoFHandler
   */
  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  /**
   * @brief Returns the current solution vector (distributed).
   * @return const reference to the distributed solution vector
   */
  const TrilinosWrappers::MPI::Vector &
  get_solution() const
  {
    return solution;
  }

  /**
   * @brief Returns the collected statistics.
   * @return const reference to RunStats
   */
  const RunStats &
  get_stats() const
  {
    return stats;
  }

  /**
   * @brief Extract solution as a serial Vector<double> for use with
   * FEFieldFunction. Useful when running on MPI_COMM_SELF (replicated
   * reference).
   */
  void get_solution_serial(Vector<double> &out) const;

  double
  compute_L2_error_against(const Function<dim> &reference_function) const;

private:
  /**
   * @brief Sets up the system (DoFs, matrices, vectors).
   *
   * Initializes:
   * - DoFHandler and degrees of freedom
   * - Constraints (hanging nodes)
   * - Sparsity patterns
   * - System matrices (Mass and Laplace)
   * - Vectors
   */
  void setup_system();

  /**
   * @brief Solves the linear system for the current time step.
   *
   * Uses CG solver with Algebraic Multigrid (AMG) preconditioner
   * to solve (M + theta*dt*A) * u_new = rhs.
   */
  void solve_time_step();

  /**
   * @brief Performs a single time step calculation.
   *
   * Assembles the right-hand side using the theta-scheme:
   * RHS = (M - (1-theta)*dt*A) * u_old + forcing_terms
   *
   * @param u_old Solution at previous time step.
   * @param dt Time step size.
   * @param t_new New time.
   * @param u_out Vector to store the new solution.
   */
  void do_time_step(const TrilinosWrappers::MPI::Vector &u_old, const double dt,
                    const double t_new, TrilinosWrappers::MPI::Vector &u_out);

  /**
   * @brief Writes the current solution to VTK files (parallel PVTU).
   */
  void output_results();

  /**
   * @brief Determines if output should be written at the current step.
   * @return true if output is due, false otherwise
   */
  bool should_write_output() const;

  /**
   * @brief Refines the mesh based on error estimation.
   *
   * Uses Kelly error estimator to flag cells for refinement and coarsening.
   * Transfers solution across mesh refinement.
   *
   * @param min_grid_level Minimum refinement level.
   * @param max_grid_level Maximum refinement level.
   */
  void refine_mesh(const unsigned int min_grid_level,
                   const unsigned int max_grid_level);

  /**
   * @brief Solve one time step using step doubling for adaptivity.
   *
   * Performs three solves (one full step, two half steps), computes error,
   * and decides whether to accept/reject the step and how to adjust dt.
   * Updates time, timestep_number, solution, and time_step on acceptance.
   *
   * @return true if step was accepted, false if rejected
   */
  bool solve_timestep_doubling();

  /**
   * @brief Solve one time step using RHS-based heuristic adaptivity.
   *
   * Estimates error from change in RHS between old and new time,
   * solves the time step, and adjusts dt based on heuristic rules.
   * Always accepts the step and updates time, timestep_number, solution.
   */
  void solve_timestep_heuristic();

  /**
   * @brief Perform adaptive mesh refinement if conditions are met.
   *
   * Checks if it's time to refine (based on timestep_number and
   * refine_every_n_steps), and if space adaptivity is enabled. If so, calls
   * refine_mesh() and updates stats.
   */
  void adapt_mesh();

  /**
   * @brief Logs mesh refinement events to CSV.
   * @param marked_refine Number of cells marked for refinement.
   * @param marked_coarsen Number of cells marked for coarsening.
   */
  void log_mesh_event(const unsigned int marked_refine,
                      const unsigned int marked_coarsen) const;

  /**
   * @brief Logs time stepping events to CSV.
   * @param t_now Current time.
   * @param dt_now Time step size.
   * @param accepted Whether the step was accepted (1) or rejected (0).
   * @param err_est Error estimate.
   * @param new_dt New suggested time step.
   */
  void log_time_event(const double t_now, const double dt_now,
                      const int accepted, const double err_est,
                      const double new_dt) const;

  /**
   * @brief Helper to compute new time step using Integral controller.
   */
  double compute_new_step_integral(const double error, const double sol_norm,
                                   const double       current_dt,
                                   const unsigned int p);

  /**
   * @brief Helper to compute new time step using PI controller (Gustafsson).
   */
  double compute_new_step_pi(const double error, const double sol_norm,
                             const double current_dt, const unsigned int p);

  // --- DEAL.II Objects (Parallel) ---
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  const FE_Q<dim>                           fe;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;

  // Parallel index sets
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // Trilinos matrices and vectors
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix laplace_matrix;
  TrilinosWrappers::SparseMatrix system_matrix;

  // Fully distributed vectors for linear algebra
  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector old_solution;
  TrilinosWrappers::MPI::Vector system_rhs;

  // Ghosted vector for output and error estimation
  TrilinosWrappers::MPI::Vector locally_relevant_solution;

  double time = 0.0;
  double time_step;
  double last_assembled_dt =
    -1.0; ///< Track last dt for which matrix was assembled
  unsigned int timestep_number = 0;

  // PI Controller state
  double previous_error = -1.0; ///< Error from previous step for PI controller

  // --- Configuration Flags ---
  bool        use_space_adaptivity;
  bool        use_time_adaptivity;
  std::string time_adaptivity_method;
  std::string time_step_controller;
  bool        use_rannacher_smoothing;

  double time_step_tolerance;
  double time_step_min;
  double time_step_max = 1e-1;
  double time_step_safety = 0.9;
  double theta;

  // --- Physical Parameters ---
  double density;              ///< rho [kg/m^3]
  double specific_heat;        ///< c_p [J/(kg K)]
  double thermal_conductivity; ///< k [W/(m K)]
  double source_intensity;     ///< Q [W/m^3]

  // --- Mesh & RHS parameters ---
  bool         use_mesh;
  int          cells_per_direction;
  unsigned int rhs_N;
  double       rhs_sigma;
  double       rhs_a;
  Point<dim>   rhs_x0;

  double end_time;

  unsigned int initial_global_refinement;
  unsigned int n_adaptive_pre_refinement_steps;
  unsigned int refine_every_n_steps;

  bool write_vtk;
  bool output_at_each_timestep; ///< Output at each timestep (debugging) or at
                                ///< fixed intervals
  double output_time_interval; ///< Time interval for output when not outputting
                               ///< each timestep
  double last_output_time = 0.0; ///< Track last time output was written

  std::string run_name;
  std::string output_dir;

  RunStats stats;

  /// Vector of (time, filename) pairs for PVD master file generation
  std::vector<std::pair<double, std::string>> times_and_names;
};
} // namespace Progetto

#endif // HEAT_EQUATION_H