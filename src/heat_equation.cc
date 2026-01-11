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
#include <deal.II/lac/solver_control.h>

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
  HeatEquation<dim>::HeatEquation(const HeatEquationParameters &params)
    : fe(1)
    , dof_handler(triangulation)
    , time_step(params.initial_time_step)
    , use_space_adaptivity(params.use_space_adaptivity)
    , use_time_adaptivity(params.use_time_adaptivity)
    , use_step_doubling(params.use_step_doubling)
    , time_step_tolerance(params.time_step_tolerance)
    , time_step_min(params.time_step_min)
    , theta(params.theta)
    , density(params.density)
    , specific_heat(params.specific_heat)
    , thermal_conductivity(params.thermal_conductivity)
    , source_intensity(params.source_intensity)
    , use_mesh(params.generate_mesh)
    , cells_per_direction(params.cells_per_direction)
    , rhs_N(params.source_frequency_N)
    , rhs_sigma(params.source_width_sigma)
    , rhs_a(params.source_magnitude_a)
    , end_time(params.end_time)
    , initial_global_refinement(params.initial_global_refinement)
    , n_adaptive_pre_refinement_steps(params.n_adaptive_pre_refinement_steps)
    , refine_every_n_steps(params.refine_every_n_steps)
    , write_vtk(params.write_vtk)
    , output_at_each_timestep(params.output_at_each_timestep)
    , output_time_interval(params.output_time_interval)
    , run_name(params.run_name)
    , output_dir(params.output_dir)
  {
    // Initialize last_output_time to negative interval so first output at t=0 will be written
    last_output_time = -output_time_interval;
    rhs_x0[0] = params.source_center_x;
    rhs_x0[1] = params.source_center_y;
  }

  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << "\n===========================================\n"
              << "[" << run_name << "] Number of active cells: " << triangulation.n_active_cells() << "\n"
              << "[" << run_name << "] Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
              << "===========================================\n";

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // Assemble Mass Matrix: M_ij = integral( phi_i * phi_j )
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
    // Multiply by Density * Specific Heat (rho * c_p)
    mass_matrix *= (density * specific_heat);

    // Assemble Laplace Matrix: A_ij = integral( grad(phi_i) * grad(phi_j) )
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
    // Multiply by Thermal Conductivity (k)
    laplace_matrix *= thermal_conductivity;

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // So that the preconditioner is initialized on first use
    last_assembled_dt = -1.0;
  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    ReductionControl         solver_control(1000, 1e-14, 1e-8);
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

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
    Vector<double> tmp(u_old.size());
    Vector<double> forcing_terms_local(u_old.size());

    mass_matrix.vmult(system_rhs, u_old);

    laplace_matrix.vmult(tmp, u_old);
    system_rhs.add(-(1.0 - theta) * dt, tmp);

    // Instantiate RightHandSide with the scaling factor Q (source_intensity)
    RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_a, rhs_x0, source_intensity);

    rhs_function.set_time(t_new);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
    forcing_terms_local = tmp;
    forcing_terms_local *= dt * theta;

    rhs_function.set_time(t_new - dt);
    VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
    forcing_terms_local.add(dt * (1.0 - theta), tmp);

    system_rhs += forcing_terms_local;

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * dt, laplace_matrix);

    constraints.condense(system_matrix, system_rhs);

    // Only reinitialize preconditioner when the matrix changes (i.e., when dt changes)
    const double dt_tolerance = 1e-14;
    if (std::abs(dt - last_assembled_dt) > dt_tolerance)
    {
      preconditioner.initialize(system_matrix, 1.0);
      last_assembled_dt = dt;
    }

    solve_time_step();

    u_out = solution;
  }

  template <int dim>
  void HeatEquation<dim>::output_results()
  {
    std::filesystem::create_directories(output_dir);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");
    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    // Use VTU format (compressed, binary) instead of VTK
    const std::string filename = "solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtu";
    std::ofstream output(output_dir + "/" + filename);
    data_out.write_vtu(output);

    // Record time and filename for PVD master file
    times_and_names.push_back({time, filename});
  }

  template <int dim>
  bool HeatEquation<dim>::should_write_output() const
  {
    if (output_at_each_timestep)
      return true;
    // Write if we've crossed an output time interval
    return (time - last_output_time) >= (output_time_interval - 1e-10);
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
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.6, 0.4);

    if (triangulation.n_levels() > 0)
    {
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() >= static_cast<int>(max_grid_level))
          cell->clear_refine_flag();

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() <= static_cast<int>(min_grid_level))
          cell->clear_coarsen_flag();
    }

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

    SolutionTransfer<dim> solution_trans(dof_handler);
    Vector<double> previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
    triangulation.execute_coarsening_and_refinement();

    setup_system();

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
  bool HeatEquation<dim>::solve_timestep_doubling()
  {
    const unsigned int p = 2;
    double dt = std::min(time_step, end_time - time);
    
    // Perform three solves: one full step, two half steps
    Vector<double> u_one(solution.size());
    Vector<double> u_two(solution.size());
    const double t_one = time + dt;

    do_time_step(old_solution, dt, t_one, u_one);
    
    const double dt2 = 0.5 * dt;
    Vector<double> u_half(solution.size());
    do_time_step(old_solution, dt2, time + dt2, u_half);
    do_time_step(u_half, dt2, t_one, u_two);

    // Compute error estimate
    Vector<double> diff(u_two);
    diff -= u_one;
    const double error = diff.l2_norm();
    const double sol_norm = std::max(1.0, u_two.l2_norm());
    const double tol_scaled = time_step_tolerance * sol_norm + 1e-16;
    const bool hit_min_dt = (dt <= time_step_min * 1.000001);

    // Decide acceptance
    if (error <= tol_scaled || hit_min_dt)
    {
      // Accept step
      time = t_one;
      ++timestep_number;
      solution = u_two;
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

      // Adjust time step for next iteration
      double err_ratio = (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
      double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
      dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
      time_step = dt_new;
      log_time_event(time, dt, 1, error, time_step);
      
      return true;
    }
    else
    {
      // Reject step and reduce time step
      double err_ratio = (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
      double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
      dt_new = std::max(dt_new, time_step_min);
      time_step = dt_new;
      std::cout << "[" << run_name << "] Rejected at t=" << time << " dt=" << dt << " err=" << error << " new_dt=" << time_step << "\n";
      stats.register_time_step_attempt(dt, false);
      log_time_event(time, dt, 0, error, time_step);
      
      return false;
    }
  }

  template <int dim>
  void HeatEquation<dim>::solve_timestep_heuristic()
  {
    double dt = std::min(time_step, end_time - time);
    const double t_new = time + dt;
    
    // Estimate error from RHS change
    Vector<double> rhs_new(solution.size());
    RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_a, rhs_x0, source_intensity);
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

    // Solve the time step (always accepted)
    Vector<double> u_new(solution.size());
    do_time_step(old_solution, dt, t_new, u_new);

    time = t_new;
    ++timestep_number;
    solution = u_new;
    old_solution = solution;

    std::cout << "[" << run_name << "] Heuristic step " << timestep_number 
              << " at t=" << time << " dt=" << dt << " err_est=" << error_est << "\n";
    stats.register_time_step_attempt(dt, true);
    stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

    // Adjust time step for next iteration based on heuristic rules
    double factor = 1.0;
    if (error_est > 2.0 * tol) 
      factor = 0.5;
    else if (error_est < 0.25 * tol) 
      factor = 2.0;

    double dt_new = dt * factor * time_step_safety;
    dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
    time_step = dt_new;
    log_time_event(time, dt, 1, error_est, time_step);
  }

  template <int dim>
  void HeatEquation<dim>::adapt_mesh()
  {
    if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
    {
      refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
      stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
    }
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
    stats.reset();
    std::filesystem::create_directories(output_dir);
    const auto t_start = std::chrono::steady_clock::now();

    if (use_mesh)
    {
      GridGenerator::subdivided_hyper_cube(triangulation, cells_per_direction);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }
    else
    {
      const std::string mesh_file_name = "../mesh/mesh-input.msh";
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream mesh_file(mesh_file_name);
      AssertThrow(mesh_file, ExcMessage("Could not open mesh file"));
      grid_in.read_msh(mesh_file);
      triangulation.refine_global(initial_global_refinement);
      setup_system();
    }

    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
    solution = old_solution;
    stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

    if (use_space_adaptivity && n_adaptive_pre_refinement_steps > 0)
    {
      for (unsigned int pre = 0; pre < n_adaptive_pre_refinement_steps; ++pre)
      {
        std::cout << "\n[" << run_name << "][Pre-refinement] step " << (pre + 1)
                  << " / " << n_adaptive_pre_refinement_steps << "\n";
        const double dt_probe = time_step;
        Vector<double> u_probe(solution.size());
        do_time_step(old_solution, dt_probe, dt_probe, u_probe);
        refine_mesh(initial_global_refinement, initial_global_refinement + n_adaptive_pre_refinement_steps);
        VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
        solution = old_solution;
        stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
      }
    }

    time = 0.0;
    timestep_number = 0;
    if (write_vtk) output_results();

    const double eps = 1e-12;

    // Main time-stepping loop
    if (!use_time_adaptivity)
    {
      // Fixed time step mode
      while (time < end_time - eps)
      {
        const double dt = std::min(time_step, end_time - time);
        const double t_new = time + dt;

        ++timestep_number;
        std::cout << "[" << run_name << "] Time step " << timestep_number
                  << " at t=" << t_new << "  dt=" << dt << "\n";

        // Solve the time step
        Vector<double> u_new(solution.size());
        do_time_step(old_solution, dt, t_new, u_new);

        time = t_new;
        solution = u_new;
        old_solution = solution;

        stats.register_time_step_attempt(dt, true);
        stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

        // Output if needed
        if (write_vtk && should_write_output())
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
      while (time < end_time - eps)
      {
        if (use_step_doubling)
        {
          // Step doubling adaptivity
          bool accepted = solve_timestep_doubling();
          
          if (accepted)
          {
            // Output if needed
            if (write_vtk && should_write_output())
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
          if (write_vtk && should_write_output())
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
    stats.cpu_seconds_total = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    std::cout << "[" << run_name << "] DONE. CPU seconds = " << stats.cpu_seconds_total << "\n";

    // Write PVD master file that links all timesteps to their physical times
    if (write_vtk && !times_and_names.empty())
    {
      std::ofstream pvd_output(output_dir + "/solution.pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
      std::cout << "[" << run_name << "] PVD master file written to " << output_dir << "/solution.pvd\n";
    }
  }

  // Explicit Instantiation for 2D
  template class HeatEquation<2>;
}