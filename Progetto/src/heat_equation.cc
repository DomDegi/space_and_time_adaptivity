/**
 * @file heat_equation.cc
 * @brief Implementation of the HeatEquation class and RunStats logic.
 */

#include "heat_equation.h"
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/solver_cg.h>

#include <iostream>
#include <fstream>
#include <chrono>

namespace Progetto
{
  using namespace dealii;

  // ==========================================================================
  // RunStats Implementation
  // ==========================================================================

  template <int dim>
  void HeatEquation<dim>::RunStats::reset() { *this = RunStats(); }

  template <int dim>
  void HeatEquation<dim>::RunStats::sample_dofs_and_cells(const unsigned int ndofs, const unsigned int ncells)
  {
    dof_min = std::min(dof_min, ndofs);
    dof_max = std::max(dof_max, ndofs);
    dof_sum += ndofs;
    dof_samples++;

    cells_min = std::min(cells_min, ncells);
    cells_max = std::max(cells_max, ncells);
  }

  template <int dim>
  void HeatEquation<dim>::RunStats::register_cg_iterations(const unsigned int it)
  {
    cg_iterations_sum += it;
    cg_iterations_min = std::min(cg_iterations_min, it);
    cg_iterations_max = std::max(cg_iterations_max, it);
  }

  template <int dim>
  void HeatEquation<dim>::RunStats::register_time_step_attempt(const double dt, const bool accepted)
  {
    n_time_steps_total++;
    if (accepted)
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
  double HeatEquation<dim>::RunStats::dt_mean() const
  {
    if (n_time_steps_accepted == 0) return 0.0;
    return dt_sum / static_cast<double>(n_time_steps_accepted);
  }

  template <int dim>
  double HeatEquation<dim>::RunStats::dof_mean() const
  {
    if (dof_samples == 0) return 0.0;
    return static_cast<double>(dof_sum) / static_cast<double>(dof_samples);
  }

  // ==========================================================================
  // HeatEquation Implementation
  // ==========================================================================

  template <int dim>
  HeatEquation<dim>::HeatEquation(const std::string &run_name_in,
                                  const std::string &output_dir_in,
                                  bool space_adaptivity,
                                  bool time_adaptivity,
                                  bool step_doubling,
                                  bool mesh,
                                  int  cells_per_direction_in,
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
                                  bool write_vtk_in)
    : fe(1) // Q1 elements (linear)
    , dof_handler(triangulation)
    , theta(theta_in)
  {
    run_name = run_name_in;
    output_dir = output_dir_in;

    use_space_adaptivity = space_adaptivity;
    use_time_adaptivity  = time_adaptivity;
    use_step_doubling    = step_doubling;

    use_mesh = mesh;
    cells_per_direction = cells_per_direction_in;

    // RHS parameters
    rhs_N = rhs_N_in;
    rhs_sigma = rhs_sigma_in;
    rhs_a     = rhs_a_in;
    rhs_x0[0] = rhs_x0_x_in;
    rhs_x0[1] = rhs_x0_y_in;

    // Physical coefficients
    material_diffusion = material_diffusion_in;
    material_mass      = material_mass_in;

    // Time parameters
    time_step_tolerance = time_step_tolerance_in;
    time_step_min       = time_step_min_in;
    end_time            = end_time_in;
    time_step           = initial_dt_in;

    // Mesh refinement parameters
    initial_global_refinement = initial_global_refinement_in;
    n_adaptive_pre_refinement_steps = n_adaptive_pre_refinement_steps_in;
    refine_every_n_steps = refine_every_n_steps_in;

    write_vtk = write_vtk_in;
  }

  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << "\n===========================================\n"
              << "[" << run_name << "] Number of active cells: " << triangulation.n_active_cells() << "\n"
              << "[" << run_name << "] Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
              << "===========================================\n";

    // Handle hanging nodes constraints (essential for adaptive meshes)
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    // Create sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    // Reinitialize matrices
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // Assemble Mass Matrix (multiplied by material_mass coefficient)
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
    mass_matrix *= material_mass;

    // Assemble Laplace/Stiffness Matrix (multiplied by diffusion coefficient)
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
    laplace_matrix *= material_diffusion;

    // Reinitialize vectors
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    const double rhs_norm = system_rhs.l2_norm();
    // Adaptive tolerance for the linear solver
    const double tol = std::max(1e-14, 1e-8 * rhs_norm);

    SolverControl            solver_control(1000, tol);
    SolverCG<Vector<double>> cg(solver_control);

    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    // Distribute constraints (hanging nodes) back to the solution vector
    constraints.distribute(solution);

    const unsigned int its = solver_control.last_step();
    stats.n_linear_solves++;
    stats.register_cg_iterations(its);
  }

  template <int dim>
  void HeatEquation<dim>::do_time_step(const Vector<double> &u_old,
                                       const double          dt,
                                       const double          t_new,
                                       Vector<double> &      u_out)
  {
    // Theta-scheme discretization:
    // (M + theta*dt*A) * U_new = (M - (1-theta)*dt*A) * U_old + dt*(theta*F_new + (1-theta)*F_old)

    Vector<double> tmp(u_old.size());
    Vector<double> forcing_terms_local(u_old.size());

    // 1. Term: M * U_old
    mass_matrix.vmult(system_rhs, u_old);

    // 2. Term: - (1 - theta) * dt * A * U_old
    laplace_matrix.vmult(tmp, u_old);
    system_rhs.add(-(1.0 - theta) * dt, tmp);

    // 3. Forcing terms (Right Hand Side)
    RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_a, rhs_x0);

    // F_new at t_new
    rhs_function.set_time(t_new);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
    forcing_terms_local = tmp;
    forcing_terms_local *= dt * theta;

    // F_old at t_old
    rhs_function.set_time(t_new - dt);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
    forcing_terms_local.add(dt * (1.0 - theta), tmp);

    system_rhs += forcing_terms_local;

    // 4. Assemble System Matrix (LHS): M + theta * dt * A
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * dt, laplace_matrix);

    // Apply boundary constraints and solve
    constraints.condense(system_matrix, system_rhs);
    solve_time_step();

    u_out = solution;
  }

  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    std::filesystem::create_directories(output_dir);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");
    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename = "solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    std::ofstream output(output_dir + "/" + filename);
    data_out.write_vtk(output);
  }

  template <int dim>
  void HeatEquation<dim>::log_mesh_event(const unsigned int marked_refine,
                                         const unsigned int marked_coarsen) const
  {
    std::filesystem::create_directories(output_dir);
    const std::string log_path = output_dir + "/mesh_log.csv";
    const bool write_header = !std::filesystem::exists(log_path);

    std::ofstream log(log_path, std::ios::app);
    if (write_header)
      log << "timestep,time,active_cells,dofs,levels,marked_refine,marked_coarsen\n";

    log << timestep_number << "," << time << ","
        << triangulation.n_active_cells() << "," << dof_handler.n_dofs() << ","
        << triangulation.n_levels() << "," << marked_refine << "," << marked_coarsen << "\n";
  }

  template <int dim>
  void HeatEquation<dim>::log_time_event(const double t_now, const double dt_now,
                                         const int accepted, const double err_est,
                                         const double new_dt) const
  {
    std::filesystem::create_directories(output_dir);
    const std::string log_path = output_dir + "/time_log.csv";
    const bool write_header = !std::filesystem::exists(log_path);

    std::ofstream log(log_path, std::ios::app);
    if (write_header)
      log << "timestep,time,dt,accepted,error_est,new_dt\n";

    log << timestep_number << "," << t_now << "," << dt_now << ","
        << accepted << "," << err_est << "," << new_dt << "\n";
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    // 1. Error Estimation (Kelly Error Estimator on the gradient jump)
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell);

    // 2. Mark cells for refinement/coarsening
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.6, 0.4);

    // 3. Enforce min/max level constraints
    if (triangulation.n_levels() > 0)
    {
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() >= static_cast<int>(max_grid_level))
          cell->clear_refine_flag();

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() <= static_cast<int>(min_grid_level))
          cell->clear_coarsen_flag();
    }

    // Count marked cells for logging
    unsigned int marked_refine = 0;
    unsigned int marked_coarsen = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->refine_flag_set()) ++marked_refine;
      if (cell->coarsen_flag_set()) ++marked_coarsen;
    }

    std::cout << "[" << run_name << "][AMR] refine_mesh at t=" << time
              << " (step " << timestep_number << ")"
              << " | active_cells=" << triangulation.n_active_cells()
              << " dofs=" << dof_handler.n_dofs()
              << " levels=" << triangulation.n_levels()
              << " | marked_refine=" << marked_refine
              << " marked_coarsen=" << marked_coarsen << "\n";

    log_mesh_event(marked_refine, marked_coarsen);

    // 4. Solution Transfer (Save old solution, refine grid, interpolate back)
    SolutionTransfer<dim> solution_trans(dof_handler);
    Vector<double> previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
    triangulation.execute_coarsening_and_refinement();

    setup_system(); // Rebuild matrices on the new mesh

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);

    old_solution.reinit(solution.size());
    old_solution = solution;
  }

  template <int dim>
  double HeatEquation<dim>::compute_L2_error_against(const Function<dim> &reference_function) const
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      reference_function,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    return difference_per_cell.l2_norm();
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
    stats.reset();
    std::filesystem::create_directories(output_dir);
    const auto t_start = std::chrono::steady_clock::now();

    // --- Mesh Initialization ---
    if (use_mesh)
    {
      GridGenerator::subdivided_hyper_cube(triangulation, cells_per_direction);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }
    else
    {
      // Load from external MSH file
      const std::string mesh_file_name = "../mesh/mesh-input.msh";
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream mesh_file(mesh_file_name);
      AssertThrow(mesh_file, ExcMessage("Could not open mesh file"));
      grid_in.read_msh(mesh_file);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }

    // --- Initial Condition ---
    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
    solution = old_solution;
    stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

    // --- Adaptive Pre-refinement ---
    // Refine the mesh at t=0 based on the initial forcing term behavior
    if (use_space_adaptivity && n_adaptive_pre_refinement_steps > 0)
    {
      for (unsigned int pre = 0; pre < n_adaptive_pre_refinement_steps; ++pre)
      {
        std::cout << "\n[" << run_name << "][Pre-refinement] step " << (pre + 1)
                  << " / " << n_adaptive_pre_refinement_steps << "\n";

        // Take a dummy small step to estimate error
        const double dt_probe = time_step;
        Vector<double> u_probe(solution.size());
        do_time_step(old_solution, dt_probe, dt_probe, u_probe);

        refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);

        // Reset solution to zero after refinement
        VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
        solution = old_solution;
        stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
      }
    }

    time = 0.0;
    timestep_number = 0;
    if (write_vtk) output_results();

    const double eps = 1e-12;

    // ========================================================================
    // Case 1: Fixed Time Stepping
    // ========================================================================
    if (!use_time_adaptivity)
    {
      while (time < end_time - eps)
      {
        const double dt = std::min(time_step, end_time - time);
        const double t_new = time + dt;

        ++timestep_number;
        std::cout << "[" << run_name << "] Time step " << timestep_number
                  << " at t=" << t_new << "  dt=" << dt << "\n";

        Vector<double> u_new(solution.size());
        do_time_step(old_solution, dt, t_new, u_new);

        time = t_new;
        solution = u_new;
        old_solution = solution;

        stats.register_time_step_attempt(dt, true);
        stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

        if (write_vtk) output_results();

        // Check for periodic space refinement
        if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
        {
          refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
          stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
        }
      }
    }
    // ========================================================================
    // Case 2: Adaptive Time Stepping
    // ========================================================================
    else
    {
      const unsigned int p = 2; // Order of convergence assumed
      while (time < end_time - eps)
      {
        double dt = std::min(time_step, end_time - time);

        if (use_step_doubling)
        {
          // --- Step Doubling Strategy ---
          // 1. Take one full step (dt)
          Vector<double> u_one(solution.size());
          Vector<double> u_two(solution.size());
          const double t_one = time + dt;
          do_time_step(old_solution, dt, t_one, u_one);

          // 2. Take two half steps (dt/2)
          const double dt2 = 0.5 * dt;
          Vector<double> u_half(solution.size());
          do_time_step(old_solution, dt2, time + dt2, u_half);
          do_time_step(u_half, dt2, t_one, u_two);

          // 3. Compute Error: ||u_full - u_two_half||
          Vector<double> diff(u_two);
          diff -= u_one;
          const double error = diff.l2_norm();
          const double sol_norm = std::max(1.0, u_two.l2_norm());
          const double tol_scaled = time_step_tolerance * sol_norm + 1e-16;

          // Force acceptance if dt is already at minimum
          const bool hit_min_dt = (dt <= time_step_min * 1.000001);

          if (error <= tol_scaled || hit_min_dt)
          {
            // ACCEPT STEP
            time = t_one;
            ++timestep_number;
            solution = u_two; // Use the more accurate solution
            old_solution = solution;

            if (hit_min_dt && error > tol_scaled)
            {
              std::cout << "\n   !!! FORCED ACCEPTANCE AT MIN_DT !!!\n"
                        << "   Time: " << time << " | Step: " << dt << " | Error: " << error << "\n";
            }
            else
            {
              std::cout << "[" << run_name << "] Time step " << timestep_number
                        << " accepted at t=" << time << "  dt=" << dt << "  error=" << error << "\n";
            }

            stats.register_time_step_attempt(dt, true);
            stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
            if (write_vtk) output_results();

            // Periodic Space Refinement
            if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
            {
              refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
              stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
            }

            // Calculate next dt
            const double err_ratio = (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
            time_step = dt_new;
            log_time_event(time, dt, 1, error, time_step);
          }
          else
          {
            // REJECT STEP
            const double err_ratio = (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::max(dt_new, time_step_min);
            time_step = dt_new;
            std::cout << "[" << run_name << "] Rejected at t=" << time << " dt=" << dt << " err=" << error << " new_dt=" << time_step << "\n";
            stats.register_time_step_attempt(dt, false);
            log_time_event(time, dt, 0, error, time_step);
          }
        }
        else
        {
          // --- Heuristic Strategy ---
          // Estimates error based on the change in the RHS (forcing term)
          const double t_new = time + dt;
          Vector<double> rhs_new(solution.size());
          RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_a, rhs_x0);
          rhs_function.set_time(t_new);
          VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, rhs_new);

          Vector<double> rhs_old(solution.size());
          rhs_function.set_time(time);
          VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, rhs_old);

          Vector<double> rhs_diff = rhs_new;
          rhs_diff -= rhs_old;
          const double rhs_diff_norm = rhs_diff.l2_norm();
          const double rhs_norm = std::max(1.0, rhs_new.l2_norm());
          const double error_est = dt * rhs_diff_norm / rhs_norm;
          const double tol = time_step_tolerance;

          Vector<double> u_new(solution.size());
          do_time_step(old_solution, dt, t_new, u_new);

          time = t_new;
          ++timestep_number;
          solution = u_new;
          old_solution = solution;

          std::cout << "[" << run_name << "] Heuristic step " << timestep_number << " at t=" << time << " dt=" << dt << " err_est=" << error_est << "\n";
          stats.register_time_step_attempt(dt, true);
          stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
          if (write_vtk) output_results();

          if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
          {
            refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
            stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
          }

          // Heuristic adjustment
          double factor = 1.0;
          if (error_est > 2.0 * tol) factor = 0.5;
          else if (error_est < 0.25 * tol) factor = 2.0;

          double dt_new = dt * factor * time_step_safety;
          dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
          time_step = dt_new;
          log_time_event(time, dt, 1, error_est, time_step);
        }
      }
    }

    const auto t_end = std::chrono::steady_clock::now();
    stats.cpu_seconds_total = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    std::cout << "[" << run_name << "] DONE. CPU seconds = " << stats.cpu_seconds_total << "\n";
  }

  // Explicit Instantiation for 2D to ensure linker finds symbols
  template class HeatEquation<2>;
}