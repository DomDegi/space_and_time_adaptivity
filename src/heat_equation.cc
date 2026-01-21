/**
 * @file heat_equation.cc
 * @brief Implementation of the HeatEquation class and RunStats logic.
 */

#include "heat_equation.h"
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Progetto
{
using namespace dealii;

// ==========================================================================
// RunStats Implementation
// ==========================================================================

template <int dim>
void
HeatEquation<dim>::RunStats::reset()
{
  *this = RunStats();
}

template <int dim>
void
HeatEquation<dim>::RunStats::sample_dofs_and_cells(const unsigned int ndofs,
                                                   const unsigned int ncells)
{
  dof_min = std::min(dof_min, ndofs);
  dof_max = std::max(dof_max, ndofs);
  dof_sum += ndofs;
  dof_samples++;

  cells_min = std::min(cells_min, ncells);
  cells_max = std::max(cells_max, ncells);
}

template <int dim>
void
HeatEquation<dim>::RunStats::register_cg_iterations(const unsigned int it)
{
  cg_iterations_sum += it;
  cg_iterations_min = std::min(cg_iterations_min, it);
  cg_iterations_max = std::max(cg_iterations_max, it);
}

template <int dim>
void
HeatEquation<dim>::RunStats::register_time_step_attempt(const double dt,
                                                        const bool   accepted)
{
  n_time_steps_total++;
  if(accepted)
    {
      n_time_steps_accepted++;
      dt_min = std::min(dt_min, dt);
      dt_max = std::max(dt_max, dt);
      dt_sum += dt;
    }
  else
    {
      n_time_steps_rejected++;
    }
}

template <int dim>
double
HeatEquation<dim>::RunStats::dt_mean() const
{
  if(n_time_steps_accepted == 0)
    return 0.0;
  return dt_sum / static_cast<double>(n_time_steps_accepted);
}

template <int dim>
double
HeatEquation<dim>::RunStats::dof_mean() const
{
  if(dof_samples == 0)
    return 0.0;
  return static_cast<double>(dof_sum) / static_cast<double>(dof_samples);
}

// ==========================================================================
// HeatEquation Implementation
// ==========================================================================

template <int dim>
HeatEquation<dim>::HeatEquation(const HeatEquationParameters &params,
                                MPI_Comm                      comm)
  : mpi_communicator(comm),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    fe(1), dof_handler(triangulation), time_step(params.initial_time_step),
    use_space_adaptivity(params.use_space_adaptivity),
    use_time_adaptivity(params.use_time_adaptivity),
    time_adaptivity_method(params.time_adaptivity_method),
    time_step_controller(params.time_step_controller),
    use_rannacher_smoothing(params.use_rannacher_smoothing),
    time_step_tolerance(params.time_step_tolerance),
    time_step_min(params.time_step_min), theta(params.theta),
    density(params.density), specific_heat(params.specific_heat),
    thermal_conductivity(params.thermal_conductivity),
    source_intensity(params.source_intensity), use_mesh(params.generate_mesh),
    cells_per_direction(params.cells_per_direction),
    rhs_N(params.source_frequency_N), rhs_sigma(params.source_width_sigma),
    rhs_a(params.source_magnitude_a), end_time(params.end_time),
    initial_global_refinement(params.initial_global_refinement),
    n_adaptive_pre_refinement_steps(params.n_adaptive_pre_refinement_steps),
    refine_every_n_steps(params.refine_every_n_steps),
    write_vtk(params.write_vtk),
    output_at_each_timestep(params.output_at_each_timestep),
    output_time_interval(params.output_time_interval),
    run_name(params.run_name), output_dir(params.output_dir)
{
  // Initialize last_output_time to negative interval so first output at t=0
  // will be written
  last_output_time = -output_time_interval;
  rhs_x0[0] = params.source_center_x;
  rhs_x0[1] = params.source_center_y;
}

template <int dim>
void
HeatEquation<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  pcout << "\n===========================================\n"
        << "[" << run_name
        << "] Number of active cells: " << triangulation.n_global_active_cells()
        << "\n"
        << "[" << run_name
        << "] Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
        << "===========================================\n";

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  SparsityTools::distribute_sparsity_pattern(
    dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                     mpi_communicator);
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                        mpi_communicator);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  old_solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                   mpi_communicator);

  // --- FIX: Pass 'constraints' to MatrixCreator ---
  // This ensures assembly respects hanging nodes and doesn't write to forbidden
  // rows

  // Assemble Mass Matrix: M_ij = integral( phi_i * phi_j )
  MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1),
                                    mass_matrix, (const Function<dim> *)nullptr,
                                    constraints);
  mass_matrix *= (density * specific_heat);

  // Assemble Laplace Matrix: A_ij = integral( grad(phi_i) * grad(phi_j) )
  MatrixCreator::create_laplace_matrix(
    dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix,
    (const Function<dim> *)nullptr, constraints);
  laplace_matrix *= thermal_conductivity;
}

template <int dim>
void
HeatEquation<dim>::solve_time_step()
{
  // Solver control with relative tolerance on RHS L2 norm
  SolverControl solver_control(solution.size(), 1e-12 * system_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

  // Algebraic Multigrid (AMG) Preconditioner (Trilinos ML/MueLu)

  TrilinosWrappers::PreconditionAMG                 preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData data;
  data.elliptic = true;
  data.n_cycles = 1;

  preconditioner.initialize(system_matrix, data);

  cg.solve(system_matrix, solution, system_rhs, preconditioner);

  // --- FIX: Correct ghost distribution for parallel AMR ---
  // 1. Update ghost values in locally_relevant_solution using solution (owned).
  locally_relevant_solution = solution;
  // 2. Resolve hanging node constraints using the ghosts.
  constraints.distribute(locally_relevant_solution);
  // 3. Copy the correctly resolved owned values back to solution.
  solution = locally_relevant_solution;

  stats.n_linear_solves++;
  stats.register_cg_iterations(solver_control.last_step());
}

template <int dim>
void
HeatEquation<dim>::do_time_step(const TrilinosWrappers::MPI::Vector &u_old,
                                const double dt, const double t_new,
                                TrilinosWrappers::MPI::Vector &u_out)
{
  system_rhs = 0;

  // Matrix contributions (these are safe as matrices are already condensed)
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

  // M * u_old
  mass_matrix.vmult(tmp, u_old);
  system_rhs.add(1.0, tmp);

  // - (1-theta) * dt * A * u_old
  laplace_matrix.vmult(tmp, u_old);
  system_rhs.add(-(1.0 - theta) * dt, tmp);

  // --- FIX: Manual Assembly of Source Term for Parallel AMR ---
  // We cannot use VectorTools::create_right_hand_side + condense because
  // condense() fails on distributed vectors when parents are ghosts.
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>     fe_values(fe, quadrature_formula,
                                update_values | update_quadrature_points |
                                  update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    RightHandSide<dim> rhs_func(rhs_N, rhs_sigma, rhs_a, rhs_x0,
                                source_intensity);

    TrilinosWrappers::MPI::Vector forcing(locally_owned_dofs, mpi_communicator);

    for(const auto &cell : dof_handler.active_cell_iterators())
      {
        if(cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_rhs = 0;

            // Force at t_new
            rhs_func.set_time(t_new);
            for(unsigned int q = 0; q < n_q_points; ++q)
              {
                const double val =
                  rhs_func.value(fe_values.quadrature_point(q));
                for(unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) += val * fe_values.shape_value(i, q) *
                                 fe_values.JxW(q) * theta * dt;
              }

            // Force at t_old
            rhs_func.set_time(t_new - dt);
            for(unsigned int q = 0; q < n_q_points; ++q)
              {
                const double val =
                  rhs_func.value(fe_values.quadrature_point(q));
                for(unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) += val * fe_values.shape_value(i, q) *
                                 fe_values.JxW(q) * (1.0 - theta) * dt;
              }

            cell->get_dof_indices(local_dof_indices);

            // This function handles constraints (adding hanging node contrib to
            // parents) AND parallel assembly (compressing contributions to
            // ghosts) correctly.
            constraints.distribute_local_to_global(cell_rhs, local_dof_indices,
                                                   forcing);
          }
      }

    forcing.compress(VectorOperation::add);
    system_rhs.add(1.0, forcing);
  }

  // --- FIX: Compress RHS before solve to finalize parallel assembly ---
  system_rhs.compress(VectorOperation::add);

  // Prepare LHS Matrix
  // LHS = M + theta * dt * A
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(theta * dt, laplace_matrix);

  // --- FIX: Compress Matrix after modifications ---
  system_matrix.compress(VectorOperation::add);

  solve_time_step();
  u_out = solution;
}

template <int dim>
void
HeatEquation<dim>::output_results()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Add solution vector (ghosted) for correct boundary visualization
  data_out.add_data_vector(locally_relevant_solution, "U");

  // Use the actual owner for each cell (avoids duplicate geometry)
  Vector<float> subdomain(triangulation.n_active_cells());
  {
    unsigned int i = 0;
    for(const auto &cell : triangulation.active_cell_iterators())
      {
        subdomain(i) = cell->subdomain_id();
        i++;
      }
  }
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

  const std::string  filename_base = "solution";
  const unsigned int padding = 4;
  data_out.write_vtu_with_pvtu_record(
    output_dir, filename_base, timestep_number, mpi_communicator,
    Utilities::MPI::n_mpi_processes(mpi_communicator), padding);

  MPI_Barrier(mpi_communicator);

  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      const std::string filename_full =
        filename_base + "_" +
        Utilities::int_to_string(timestep_number, padding) + ".pvtu";
      if(times_and_names.empty() ||
         std::abs(times_and_names.back().first - time) > 1e-9)
        times_and_names.push_back({time, filename_full});

      std::ofstream pvd_output(output_dir + "solution.pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
    }
}

template <int dim>
bool
HeatEquation<dim>::should_write_output() const
{
  if(output_at_each_timestep)
    return true;
  // Write if we've crossed an output time interval
  return (time - last_output_time) >= (output_time_interval - 1e-10);
}

template <int dim>
void
HeatEquation<dim>::log_mesh_event(const unsigned int marked_refine,
                                  const unsigned int marked_coarsen) const
{
  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::filesystem::create_directories(output_dir);
      const std::string log_path = output_dir + "mesh_log.csv";
      const bool        write_header = !std::filesystem::exists(log_path);

      std::ofstream log(log_path, std::ios::app);
      if(write_header)
        log << "timestep,time,active_cells,dofs,levels,marked_refine,marked_"
               "coarsen\n";

      log << timestep_number << "," << time << ","
          << triangulation.n_global_active_cells() << ","
          << dof_handler.n_dofs() << "," << triangulation.n_levels() << ","
          << marked_refine << "," << marked_coarsen << "\n";
    }
}

template <int dim>
void
HeatEquation<dim>::log_time_event(const double t_now, const double dt_now,
                                  const int accepted, const double err_est,
                                  const double new_dt) const
{
  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::filesystem::create_directories(output_dir);
      const std::string log_path = output_dir + "time_log.csv";
      const bool        write_header = !std::filesystem::exists(log_path);

      std::ofstream log(log_path, std::ios::app);
      if(write_header)
        log << "timestep,time,dt,accepted,error_est,new_dt\n";

      log << timestep_number << "," << t_now << "," << dt_now << "," << accepted
          << "," << err_est << "," << new_dt << "\n";
    }
}

template <int dim>
double
HeatEquation<dim>::compute_new_step_integral(const double       error,
                                             const double       sol_norm,
                                             const double       current_dt,
                                             const unsigned int p)
{
  double err_ratio =
    (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
  double dt_new =
    current_dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
  return dt_new;
}

template <int dim>
double
HeatEquation<dim>::compute_new_step_pi(const double       error,
                                       const double       sol_norm,
                                       const double       current_dt,
                                       const unsigned int p)
{
  // Gustafsson et al. PI Control
  const double order = p;
  const double k_I = 0.3 / (order + 1.0); // Integral gain
  const double k_P = 0.4 / (order + 1.0); // Proportional gain

  double factor = 1.0;
  double scaled_err =
    (error > 0.0) ? (error / (time_step_tolerance * sol_norm)) : 1e-6;

  // Safety check for zero error
  if(scaled_err < 1e-10)
    scaled_err = 1e-10;

  if(previous_error == -1.0)
    {
      // First step or after rejection: Integral controller only
      factor =
        time_step_safety * std::pow(1.0 / scaled_err, 1.0 / (order + 1.0));
    }
  else
    {
      // PI Controller
      double scaled_prev_err =
        (previous_error > 0.0)
          ? (previous_error / (time_step_tolerance * sol_norm))
          : 1e-6;
      // Prevent division by zero or extreme values
      if(scaled_prev_err < 1e-10)
        scaled_prev_err = 1e-10;

      const double ratio_now = 1.0 / scaled_err;
      const double ratio_prev =
        scaled_prev_err; // (1/e_n) / (1/e_{n-1}) = e_{n-1}/e_n

      factor = time_step_safety * std::pow(ratio_now, k_I) *
               std::pow(ratio_now * ratio_prev, k_P);
    }

  return current_dt * factor;
}

template <int dim>
void
HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                               const unsigned int max_grid_level)
{
  // --- FIX: Ensure ghosts are up-to-date before error estimation ---
  locally_relevant_solution = solution;
  constraints.distribute(locally_relevant_solution);

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
    dof_handler, QGauss<dim - 1>(fe.degree + 1),
    std::map<types::boundary_id, const Function<dim> *>(),
    locally_relevant_solution, estimated_error_per_cell, ComponentMask(),
    nullptr, 0, triangulation.locally_owned_subdomain());

  // Use parallel distributed refinement
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation, estimated_error_per_cell, 0.3, 0.03);

  if(triangulation.n_levels() > 0)
    {
      for(const auto &cell : triangulation.active_cell_iterators())
        if(cell->level() >= static_cast<int>(max_grid_level))
          cell->clear_refine_flag();

      for(const auto &cell : triangulation.active_cell_iterators())
        if(cell->level() <= static_cast<int>(min_grid_level))
          cell->clear_coarsen_flag();
    }

  unsigned int marked_refine = 0;
  unsigned int marked_coarsen = 0;
  for(const auto &cell : triangulation.active_cell_iterators())
    {
      if(cell->refine_flag_set())
        ++marked_refine;
      if(cell->coarsen_flag_set())
        ++marked_coarsen;
    }

  pcout << "[" << run_name << "][AMR] refine_mesh at t=" << time << " (step "
        << timestep_number << ")"
        << " | active_cells=" << triangulation.n_global_active_cells()
        << " dofs=" << dof_handler.n_dofs()
        << " levels=" << triangulation.n_levels()
        << " | marked_refine=" << marked_refine
        << " marked_coarsen=" << marked_coarsen << "\n";

  log_mesh_event(marked_refine, marked_coarsen);

  // Use parallel solution transfer
  parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
    solution_trans(dof_handler);

  // Prepare ghosted vectors for reliable data transfer at boundaries
  // locally_relevant_solution is already ghosted, ensure it's up to date
  locally_relevant_solution = solution;
  constraints.distribute(locally_relevant_solution);

  // We need a ghosted vector for old_solution too
  TrilinosWrappers::MPI::Vector locally_relevant_old_solution(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  locally_relevant_old_solution = old_solution;
  constraints.distribute(locally_relevant_old_solution);

  // Pass fully distributed (ghosted) vectors
  std::vector<const TrilinosWrappers::MPI::Vector *> x_system(2);
  x_system[0] = &locally_relevant_solution;
  x_system[1] = &locally_relevant_old_solution;
  solution_trans.prepare_for_coarsening_and_refinement(x_system);

  triangulation.execute_coarsening_and_refinement();

  setup_system();

  // Recover vectors
  std::vector<TrilinosWrappers::MPI::Vector *> x_tmp(2);
  TrilinosWrappers::MPI::Vector new_sol(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector new_old_sol(locally_owned_dofs,
                                            mpi_communicator);
  x_tmp[0] = &new_sol;
  x_tmp[1] = &new_old_sol;

  solution_trans.interpolate(x_tmp);

  // --- FIX: Correct constraint distribution for solution transfer ---
  // We must use the ghosted vector locally_relevant_solution to satisfy
  // constraints that depend on ghost values (parents).

  // 1. Fix 'solution'
  solution = new_sol;
  locally_relevant_solution = solution; // Communicate owned values to ghosts
  constraints.distribute(locally_relevant_solution); // Resolve constraints
  solution = locally_relevant_solution; // Copy back resolved owned values

  // 2. Fix 'old_solution'
  old_solution = new_old_sol;
  locally_relevant_solution = old_solution; // Reuse ghosted vector for swap
  constraints.distribute(locally_relevant_solution);
  old_solution = locally_relevant_solution;

  // Restore locally_relevant_solution to current solution for output/estimators
  locally_relevant_solution = solution;

  // Update ghosted vector
  locally_relevant_solution = solution;
}

template <int dim>
void
HeatEquation<dim>::get_solution_serial(Vector<double> &out) const
{
  // Copy the distributed Trilinos vector to a serial Vector<double>
  // This works correctly when running on MPI_COMM_SELF (replicated reference)
  out.reinit(solution.size());
  for(unsigned int i = 0; i < solution.size(); ++i)
    {
      out(i) = solution(i);
    }
}

template <int dim>
double
HeatEquation<dim>::compute_L2_error_against(
  const Function<dim> &reference_function) const
{
  Vector<float> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(
    dof_handler, solution, reference_function, difference_per_cell,
    QGauss<dim>(fe.degree + 2), VectorTools::L2_norm);
  return difference_per_cell.l2_norm();
}

template <int dim>
bool
HeatEquation<dim>::solve_timestep_doubling()
{
  const unsigned int p = 2;
  double             dt = std::min(time_step, end_time - time);

  // Perform three solves: one full step, two half steps
  TrilinosWrappers::MPI::Vector u_one(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector u_two(locally_owned_dofs, mpi_communicator);
  const double                  t_one = time + dt;

  do_time_step(old_solution, dt, t_one, u_one);

  const double                  dt2 = 0.5 * dt;
  TrilinosWrappers::MPI::Vector u_half(locally_owned_dofs, mpi_communicator);
  do_time_step(old_solution, dt2, time + dt2, u_half);
  do_time_step(u_half, dt2, t_one, u_two);

  // Compute error estimate
  TrilinosWrappers::MPI::Vector diff = u_two;
  diff -= u_one;
  const double error =
    diff.l2_norm(); // Trilinos handles MPI reduction automatically
  const double sol_norm = std::max(1.0, u_two.l2_norm());
  const double tol_scaled = time_step_tolerance * sol_norm + 1e-16;
  const bool   hit_min_dt = (dt <= time_step_min * 1.000001);

  // Decide acceptance
  if(error <= tol_scaled || hit_min_dt)
    {
      // Accept step
      time = t_one;
      ++timestep_number;
      solution = u_two;
      old_solution = solution;
      locally_relevant_solution = solution;

      if(hit_min_dt && error > tol_scaled)
        {
          pcout << "\n   !!! FORCED ACCEPTANCE AT MIN_DT !!!\n"
                << "   Time: " << time << " | Step: " << dt
                << " | Error: " << error << "\n";
        }
      else
        {
          pcout << "[" << run_name << "] Time step " << timestep_number
                << " accepted at t=" << time << "  dt=" << dt
                << "  error=" << error << "\n";
        }

      stats.register_time_step_attempt(dt, true);
      stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                                  triangulation.n_global_active_cells());

      // Adjust time step for next iteration
      double dt_new;

      if(time_step_controller == "pi")
        {
          dt_new = compute_new_step_pi(error, sol_norm, dt, p);
        }
      else
        {
          dt_new = compute_new_step_integral(error, sol_norm, dt, p);
        }

      dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);

      // Limit growth/shrink to prevent wild oscillations
      // dt_new = std::min(dt_new, 2.0 * dt); // Optional stability clamp

      time_step = dt_new;
      previous_error = error; // Store for next step

      log_time_event(time, dt, 1, error, time_step);

      return true;
    }
  else
    {
      // Reject step and reduce time step
      // For rejection, always use Integral controller (safer)
      double dt_new = compute_new_step_integral(error, sol_norm, dt, p);

      dt_new = std::max(dt_new, time_step_min);
      time_step = dt_new;

      // Reset PI controller history on rejection
      previous_error = -1.0;

      pcout << "[" << run_name << "] Rejected at t=" << time << " dt=" << dt
            << " err=" << error << " new_dt=" << time_step << "\n";
      stats.register_time_step_attempt(dt, false);
      log_time_event(time, dt, 0, error, time_step);

      return false;
    }
}

template <int dim>
void
HeatEquation<dim>::solve_timestep_heuristic()
{
  double       dt = std::min(time_step, end_time - time);
  const double t_new = time + dt;

  // Estimate error from RHS change
  TrilinosWrappers::MPI::Vector rhs_new(locally_owned_dofs, mpi_communicator);
  RightHandSide<dim>            rhs_function(rhs_N, rhs_sigma, rhs_a, rhs_x0,
                                             source_intensity);
  rhs_function.set_time(t_new);
  VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1),
                                      rhs_function, rhs_new);

  TrilinosWrappers::MPI::Vector rhs_old(locally_owned_dofs, mpi_communicator);
  rhs_function.set_time(time);
  VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1),
                                      rhs_function, rhs_old);

  TrilinosWrappers::MPI::Vector rhs_diff = rhs_new;
  rhs_diff -= rhs_old;
  const double rhs_diff_norm = rhs_diff.l2_norm();
  const double rhs_norm = std::max(1.0, rhs_new.l2_norm());
  const double error_est = dt * rhs_diff_norm / rhs_norm;
  const double tol = time_step_tolerance;

  // Solve the time step (always accepted)
  TrilinosWrappers::MPI::Vector u_new(locally_owned_dofs, mpi_communicator);
  do_time_step(old_solution, dt, t_new, u_new);

  time = t_new;
  ++timestep_number;
  solution = u_new;
  old_solution = solution;
  locally_relevant_solution = solution;

  pcout << "[" << run_name << "] Heuristic step " << timestep_number
        << " at t=" << time << " dt=" << dt << " err_est=" << error_est << "\n";
  stats.register_time_step_attempt(dt, true);
  stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                              triangulation.n_global_active_cells());

  // Adjust time step for next iteration based on heuristic rules
  double factor = 1.0;
  if(error_est > 2.0 * tol)
    factor = 0.5;
  else if(error_est < 0.25 * tol)
    factor = 2.0;

  double dt_new = dt * factor * time_step_safety;
  dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
  time_step = dt_new;
  log_time_event(time, dt, 1, error_est, time_step);
}

template <int dim>
void
HeatEquation<dim>::adapt_mesh()
{
  if((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
    {
      refine_mesh(initial_global_refinement,
                  initial_global_refinement + n_adaptive_pre_refinement_steps);
      stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                                  triangulation.n_global_active_cells());
    }
}

template <int dim>
void
HeatEquation<dim>::run()
{
  const double user_configured_theta = theta; // Backup user setting

  stats.reset();
  const auto t_start = std::chrono::steady_clock::now();

  if(use_mesh)
    {
      GridGenerator::subdivided_hyper_cube(triangulation, cells_per_direction);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }
  else
    {
      const std::string mesh_file_name = "../mesh/mesh-input.msh";
      GridIn<dim>       grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream mesh_file(mesh_file_name);
      AssertThrow(mesh_file, ExcMessage("Could not open mesh file"));
      grid_in.read_msh(mesh_file);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }

  VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(),
                           old_solution);
  solution = old_solution;
  locally_relevant_solution = solution;
  stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                              triangulation.n_global_active_cells());

  // --- FIX: Ensure output directory exists before any rank attempts to write
  // ---
  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::filesystem::create_directories(output_dir);
    }
  // Synchronize to ensure directory exists before continuing
  MPI_Barrier(mpi_communicator);
  // ---------------------------------------------------------------------------

  if(use_space_adaptivity && n_adaptive_pre_refinement_steps > 0)
    {
      for(unsigned int pre = 0; pre < n_adaptive_pre_refinement_steps; ++pre)
        {
          pcout << "\n[" << run_name << "][Pre-refinement] step " << (pre + 1)
                << " / " << n_adaptive_pre_refinement_steps << "\n";
          const double                  dt_probe = time_step;
          TrilinosWrappers::MPI::Vector u_probe(locally_owned_dofs,
                                                mpi_communicator);
          do_time_step(old_solution, dt_probe, dt_probe, u_probe);
          refine_mesh(initial_global_refinement,
                      initial_global_refinement +
                        n_adaptive_pre_refinement_steps);
          VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(),
                                   old_solution);
          solution = old_solution;
          locally_relevant_solution = solution;
          stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                                      triangulation.n_global_active_cells());
        }
    }

  time = 0.0;
  timestep_number = 0;
  if(write_vtk)
    output_results();

  const double eps = 1e-12;

  // Main time-stepping loop
  if(!use_time_adaptivity)
    {
      // Fixed time step mode
      while(time < end_time - eps)
        {
          const double dt = std::min(time_step, end_time - time);
          // Rannacher Smoothing (Fixed Step): First few steps use Implicit
          // Euler if enabled
          if(use_rannacher_smoothing && timestep_number < 4 &&
             std::abs(user_configured_theta - 1.0) > 1e-9)
            theta = 1.0;
          else
            theta = user_configured_theta;

          const double t_new = time + dt;

          ++timestep_number;
          pcout << "[" << run_name << "] Time step " << timestep_number
                << " at t=" << t_new << "  dt=" << dt << "\n";

          // Solve the time step
          TrilinosWrappers::MPI::Vector u_new(locally_owned_dofs,
                                              mpi_communicator);
          do_time_step(old_solution, dt, t_new, u_new);

          time = t_new;
          solution = u_new;
          old_solution = solution;
          locally_relevant_solution = solution;

          stats.register_time_step_attempt(dt, true);
          stats.sample_dofs_and_cells(dof_handler.n_dofs(),
                                      triangulation.n_global_active_cells());

          MPI_Barrier(mpi_communicator);

          // Output if needed
          if(write_vtk && should_write_output())
            {
              output_results();
              last_output_time = time;
            }

          // Adapt mesh if needed
          adapt_mesh();
        }
    }
  else
    {
      // Adaptive time stepping mode
      while(time < end_time - eps)
        {
          if(time_adaptivity_method == "step_doubling")
            {
              // Step doubling adaptivity

              // Rannacher Smoothing (Adaptive) if enabled
              if(use_rannacher_smoothing && timestep_number < 4 &&
                 std::abs(user_configured_theta - 1.0) > 1e-9)
                theta = 1.0;
              else
                theta = user_configured_theta;

              bool accepted = solve_timestep_doubling();

              if(accepted)
                {
                  // Output if needed
                  if(write_vtk && should_write_output())
                    {
                      output_results();
                      last_output_time = time;
                    }

                  // Adapt mesh if needed
                  adapt_mesh();
                }
            }
          else
            {
              // Heuristic adaptivity
              solve_timestep_heuristic();

              // Output if needed
              if(write_vtk && should_write_output())
                {
                  output_results();
                  last_output_time = time;
                }

              // Adapt mesh if needed
              adapt_mesh();
            }
        }
    }

  const auto t_end = std::chrono::steady_clock::now();
  stats.cpu_seconds_total =
    std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start)
      .count();
  pcout << "[" << run_name
        << "] DONE. CPU seconds = " << stats.cpu_seconds_total << "\n";

  // Write PVD master file that links all timesteps to their physical times
  // (only on rank 0)
  if(write_vtk && !times_and_names.empty() &&
     Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::ofstream pvd_output(output_dir + "solution.pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
      pcout << "[" << run_name << "] PVD master file written to " << output_dir
            << "/solution.pvd\n";
    }
}

// Explicit Instantiation for 2D
template class HeatEquation<2>;
} // namespace Progetto