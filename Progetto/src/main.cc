/**
 * @file main.cc
 * @brief Main entry point for the Heat Equation simulation project.
 *
 * Handles user input, parameter configuration, and executes the simulation
 * in various modes (Fixed/Adaptive Space/Time).
 */

#include "heat_equation.h"
#include "utilities.h"
#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>

/**
 * @brief Appends simulation statistics to a central CSV file for comparison.
 * * @param path Path to the CSV file.
 * @param config_name Label for the current run configuration.
 * @param l2_error_T The final L2 error against the reference solution.
 * @param st The statistics structure gathered during the run.
 */
static void append_comparison_csv(const std::string &path,
                                  const std::string &config_name,
                                  const double l2_error_T,
                                  const Progetto::HeatEquation<2>::RunStats &st)
{
  const bool write_header = !std::filesystem::exists(path);

  std::ofstream out(path, std::ios::app);
  if (write_header)
  {
    out << "config,l2_error_T,cpu_s,"
           "n_linear_solves,cg_iters_sum,cg_iters_min,cg_iters_max,"
           "n_steps_total,n_steps_accepted,n_steps_rejected,dt_min,dt_max,dt_mean,"
           "dof_min,dof_max,dof_mean,cells_min,cells_max\n";
  }

  const double dt_min = (st.n_time_steps_accepted > 0) ? st.dt_min : 0.0;
  const unsigned int cg_min = (st.cg_iterations_min == std::numeric_limits<unsigned int>::max()) ? 0u : st.cg_iterations_min;
  const unsigned int dof_min = (st.dof_min == std::numeric_limits<unsigned int>::max()) ? 0u : st.dof_min;
  const unsigned int cells_min = (st.cells_min == std::numeric_limits<unsigned int>::max()) ? 0u : st.cells_min;

  out << config_name << ","
      << l2_error_T << ","
      << st.cpu_seconds_total << ","
      << st.n_linear_solves << ","
      << st.cg_iterations_sum << ","
      << cg_min << ","
      << st.cg_iterations_max << ","
      << st.n_time_steps_total << ","
      << st.n_time_steps_accepted << ","
      << st.n_time_steps_rejected << ","
      << dt_min << ","
      << st.dt_max << ","
      << st.dt_mean() << ","
      << dof_min << ","
      << st.dof_max << ","
      << st.dof_mean() << ","
      << cells_min << ","
      << st.cells_max
      << "\n";
}

int main()
{
  try
  {
    clear_solutions_folder();
    using namespace Progetto;

    // --- Grid / Mesh Input ---
    const bool mesh = ask_bool("Do you want to generate a grid inside the program? (answer 'n' to load from file)");
    int cells_per_direction = 0;

    if (mesh)
    {
      std::cout << "Enter the number of cells per direction (positive integer): ";
      std::cin >> cells_per_direction;
      if (cells_per_direction <= 0)
      {
        std::cerr << "Invalid number of cells. It must be a positive integer.\n";
        return 1;
      }
      std::cout << "Generating a mesh with " << cells_per_direction << " cells per direction.\n";
    }

    // --- Equation Physical Parameters ---
    const double T_end       = ask_double_default("Enter T (final time)", 0.5);
    const double sigma       = ask_double_default("Enter source width sigma", 0.5);
    const double x0_x        = ask_double_default("Enter source center x0_x", 0.5);
    const double x0_y        = ask_double_default("Enter source center x0_y", 0.5);
    const unsigned int N_val = ask_uint_default("Enter N (source frequency term)", 5);
    const double a_val       = ask_double_default("Enter parameter 'A' (oscillation magnitude)", 5.0);

    // 1. Ask for Physical Coefficients
    double user_diffusion = 1.0;
    double user_mass      = 1.0;

    const bool change_physics = ask_bool("Do you want to customize physical material coefficients (Diffusion/Reaction)?");
    if (change_physics)
    {
      std::cout << "--- Material Properties ---\n";
      user_diffusion = ask_double_default("Enter Diffusion coefficient (k)", 1.0);
      user_mass      = ask_double_default("Enter Reaction/Mass coefficient (c)", 1.0);
      std::cout << "---------------------------\n";
    }
    else
    {
      std::cout << "Using default material coefficients (k=1.0, c=1.0).\n";
    }

    // 2. Ask for Solver / Time Adaptivity Settings
    double user_theta = 0.5;
    double user_tol   = 1e-6;
    double user_dt_min = 1e-4;

    const bool change_solver = ask_bool("Do you want to customize solver & time adaptivity settings (Theta, Tolerance, Min DT)?");
    if (change_solver)
    {
        std::cout << "--- Solver Settings ---\n"
                  << "Defaults: Theta=0.5 (Crank-Nicolson), Tolerance=1e-6, Min DT=1e-4.\n";
        user_theta = ask_double_default("Enter Theta (0.0=Explicit Euler, 0.5=Crank-Nicolson, 1.0=Implicit Euler)", 0.5);
        user_tol   = ask_double_default("Enter Time Step Tolerance", 1e-6);
        user_dt_min = ask_double_default("Enter Minimum Time Step (dt_min)", 1e-4);
        std::cout << "-----------------------\n";
    }
    else
    {
        std::cout << "Using default solver settings (Theta=0.5, Tol=1e-6, dt_min=1e-4).\n";
    }

    const bool time_step_doubling = ask_bool("Time adaptivity: use step-doubling? (answer 'n' to use simple heuristic)");

    // --- Simulation Mode Selection ---
    std::cout << "\n============================================\n"
              << "          SELECT SIMULATION MODE            \n"
              << "============================================\n"
              << "0 - Full Comparison (Run Reference + all 4 configurations) [Default]\n"
              << "1 - Fixed Space + Fixed Time\n"
              << "2 - Adaptive Space + Fixed Time\n"
              << "3 - Fixed Space + Adaptive Time\n"
              << "4 - Adaptive Space + Adaptive Time\n"
              << "--------------------------------------------\n";

    unsigned int run_mode = ask_uint_default("Enter selection", 0);
    if (run_mode > 4) run_mode = 0;

    // --- Hardcoded Internal Parameters (Calibration) ---
    const double dt0 = 1.0 / 500.0;
    const unsigned int base_refine = 2;
    const unsigned int pre_steps   = 4;
    const unsigned int refine_every = 5;
    const bool write_vtk = true;

    // Reference solver parameters (higher resolution)
    const double dt_ref = dt0 / 10.0;
    const unsigned int ref_refine = base_refine + 2;

    std::unique_ptr<HeatEquation<2>> reference_solver;
    std::unique_ptr<dealii::Functions::FEFieldFunction<2>> reference_function;

    // Determine if Reference Run is needed
    bool run_reference = false;
    if (run_mode == 0) run_reference = true;
    else run_reference = ask_bool("Do you want to run the High-Resolution Reference solver first (to compute L2 errors)?");

    if (run_reference)
    {
      int ref_cells = cells_per_direction;
      if (mesh) ref_cells = static_cast<int>(std::max(1, cells_per_direction * 2));

      reference_solver = std::make_unique<HeatEquation<2>>(
        "reference", "solutions/reference",
        false, false, false, mesh, ref_cells,
        N_val, sigma, a_val, x0_x, x0_y,
        user_diffusion, user_mass, user_theta, user_tol, user_dt_min,
        T_end, dt_ref, ref_refine, 0, refine_every, false);

      std::cout << "\n========== RUN REFERENCE (fixed mesh + fixed dt) ==========\n";
      reference_solver->run();

      // Create a function wrapper around the reference solution for error computation
      reference_function = std::make_unique<dealii::Functions::FEFieldFunction<2>>(
        reference_solver->get_dof_handler(), reference_solver->get_solution());
      reference_function->set_time(T_end);
    }
    else
    {
      std::cout << "\nSkipping reference solver. L2 errors will be NaN.\n";
    }

    const std::string summary_path = "solutions/summary_comparison.csv";

    // Lambda to run a specific configuration and log results
    auto run_one = [&](const std::string &name, bool use_space, bool use_time, bool use_sd)
    {
      std::string outdir = "solutions/" + name;
      HeatEquation<2> solver(
        name, outdir, use_space, use_time, use_sd, mesh, cells_per_direction,
        N_val, sigma, a_val, x0_x, x0_y,
        user_diffusion, user_mass, user_theta, user_tol, user_dt_min,
        T_end, dt0, base_refine, pre_steps, refine_every, write_vtk);

      std::cout << "\n========== RUN " << name << " ==========\n";
      solver.run();

      double l2_err = std::numeric_limits<double>::quiet_NaN();
      if (reference_function)
        l2_err = solver.compute_L2_error_against(*reference_function);

      std::cout << "[" << name << "] L2 error at T = " << l2_err << "\n";
      append_comparison_csv(summary_path, name, l2_err, solver.get_stats());
    };

    // Execute based on selected mode
    if (run_mode == 0)
    {
       run_one("fixed_space_fixed_time", false, false, false);
       run_one("adaptive_space_fixed_time", true,  false, false);
       run_one("fixed_space_adaptive_time", false, true,  time_step_doubling);
       run_one("adaptive_space_adaptive_time", true,  true,  time_step_doubling);
       std::cout << "\nComparison finished. CSV saved to: " << summary_path << "\n";
    }
    else if (run_mode == 1) run_one("fixed_space_fixed_time", false, false, false);
    else if (run_mode == 2) run_one("adaptive_space_fixed_time", true,  false, false);
    else if (run_mode == 3) run_one("fixed_space_adaptive_time", false, true,  time_step_doubling);
    else if (run_mode == 4) run_one("adaptive_space_adaptive_time", true,  true,  time_step_doubling);

    return 0;
  }
  catch (std::exception &exc)
  {
    std::cerr << "\n\n----------------------------------------------------\n"
              << "Exception on processing:\n" << exc.what() << "\nAborting!\n"
              << "----------------------------------------------------\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "\n\n----------------------------------------------------\n"
              << "Unknown exception!\nAborting!\n"
              << "----------------------------------------------------\n";
    return 1;
  }
}