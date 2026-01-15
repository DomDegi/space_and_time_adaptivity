/**
 * @file main.cc
 * @brief Main entry point for the Heat Equation simulation project.
 *
 * Reads configuration from a parameter file using dealii::ParameterHandler
 * and executes the simulation in various modes (Fixed/Adaptive Space/Time).
 */

#include "heat_equation.h"
#include "utilities.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/fe_field_function.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

/**
 * @brief Appends simulation statistics to a central CSV file for comparison.
 */
static void
append_comparison_csv(const std::string &path, const std::string &config_name,
                      const double                               l2_error_T,
                      const Progetto::HeatEquation<2>::RunStats &st)
{
  const bool write_header = !std::filesystem::exists(path);

  std::ofstream out(path, std::ios::app);
  if(write_header)
    {
      out << "config,l2_error_T,cpu_s,"
             "n_linear_solves,cg_iters_sum,cg_iters_min,cg_iters_max,"
             "n_steps_total,n_steps_accepted,n_steps_rejected,dt_min,dt_max,dt_"
             "mean,"
             "dof_min,dof_max,dof_mean,cells_min,cells_max\n";
    }

  const double       dt_min = (st.n_time_steps_accepted > 0) ? st.dt_min : 0.0;
  const unsigned int cg_min =
    (st.cg_iterations_min == std::numeric_limits<unsigned int>::max())
      ? 0u
      : st.cg_iterations_min;
  const unsigned int dof_min =
    (st.dof_min == std::numeric_limits<unsigned int>::max()) ? 0u : st.dof_min;
  const unsigned int cells_min =
    (st.cells_min == std::numeric_limits<unsigned int>::max()) ? 0u
                                                               : st.cells_min;

  out << config_name << "," << l2_error_T << "," << st.cpu_seconds_total << ","
      << st.n_linear_solves << "," << st.cg_iterations_sum << "," << cg_min
      << "," << st.cg_iterations_max << "," << st.n_time_steps_total << ","
      << st.n_time_steps_accepted << "," << st.n_time_steps_rejected << ","
      << dt_min << "," << st.dt_max << "," << st.dt_mean() << "," << dof_min
      << "," << st.dof_max << "," << st.dof_mean() << "," << cells_min << ","
      << st.cells_max << "\n";
}

/**
 * @brief Declares all parameters for the ParameterHandler.
 */
static void
declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Mesh");
  {
    prm.declare_entry(
      "generate_mesh", "true", dealii::Patterns::Bool(),
      "Generate mesh internally (true) or load from file (false)");
    prm.declare_entry("cells_per_direction", "5", dealii::Patterns::Integer(1),
                      "Number of cells per direction for generated mesh");
  }
  prm.leave_subsection();

  prm.enter_subsection("Physical Parameters");
  {
    prm.declare_entry("final_time", "0.5", dealii::Patterns::Double(0.0),
                      "Final simulation time T");
    prm.declare_entry("source_width_sigma", "0.5",
                      dealii::Patterns::Double(0.0),
                      "Source width parameter sigma (must be positive)");
    prm.declare_entry("source_center_x", "0.5", dealii::Patterns::Double(),
                      "Source center x-coordinate");
    prm.declare_entry("source_center_y", "0.5", dealii::Patterns::Double(),
                      "Source center y-coordinate");
    prm.declare_entry("source_frequency_N", "5", dealii::Patterns::Integer(0),
                      "Source frequency term N");
    prm.declare_entry("oscillation_magnitude_A", "5.0",
                      dealii::Patterns::Double(),
                      "Oscillation magnitude parameter A");
  }
  prm.leave_subsection();

  prm.enter_subsection("Material Properties");
  {
    prm.declare_entry("density", "1.0", dealii::Patterns::Double(0.0),
                      "Density rho [kg/m^3] (must be positive)");
    prm.declare_entry("specific_heat", "1.0", dealii::Patterns::Double(0.0),
                      "Specific heat c_p [J/(kg K)] (must be positive)");
    prm.declare_entry("thermal_conductivity", "1.0",
                      dealii::Patterns::Double(0.0),
                      "Thermal conductivity k [W/(m K)] (must be positive)");
    prm.declare_entry("source_intensity", "1.0", dealii::Patterns::Double(),
                      "Source intensity Q [W/m^3]");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solver Settings");
  {
    prm.declare_entry(
      "theta", "0.5", dealii::Patterns::Double(0.0, 1.0),
      "Theta scheme parameter (0=Explicit, 0.5=Crank-Nicolson, 1=Implicit)");
    prm.declare_entry("time_step_tolerance", "1e-6",
                      dealii::Patterns::Double(0.0),
                      "Time step adaptivity tolerance");
    prm.declare_entry("minimum_time_step", "1e-4",
                      dealii::Patterns::Double(0.0),
                      "Minimum allowed time step");
    prm.declare_entry("use_step_doubling", "true", dealii::Patterns::Bool(),
                      "Use step-doubling for time adaptivity (true) or simple "
                      "heuristic (false)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Simulation Control");
  {
    prm.declare_entry(
      "run_mode", "0", dealii::Patterns::Integer(0, 4),
      "Simulation mode: 0=Full comparison, 1=Fixed/Fixed, 2=Adaptive/Fixed, "
      "3=Fixed/Adaptive, 4=Adaptive/Adaptive");
    prm.declare_entry(
      "run_reference", "true", dealii::Patterns::Bool(),
      "Run high-resolution reference solver for L2 error computation");
    prm.declare_entry("initial_time_step", "0.002",
                      dealii::Patterns::Double(0.0), "Initial time step size");
    prm.declare_entry("base_refinement", "2", dealii::Patterns::Integer(0),
                      "Base global refinement level");
    prm.declare_entry("pre_refinement_steps", "4", dealii::Patterns::Integer(0),
                      "Number of adaptive pre-refinement steps");
    prm.declare_entry("refine_every_n_steps", "5", dealii::Patterns::Integer(1),
                      "Refine mesh every N time steps");
    prm.declare_entry("write_vtk", "true", dealii::Patterns::Bool(),
                      "Write VTK output files");
    prm.declare_entry("output_at_each_timestep", "false",
                      dealii::Patterns::Bool(),
                      "Output at each solver timestep (true for debugging, "
                      "false for fixed time intervals)");
    prm.declare_entry(
      "output_time_interval", "0.002", dealii::Patterns::Double(0.0),
      "Time interval for VTK output when output_at_each_timestep is false");
  }
  prm.leave_subsection();
}

int
main(int argc, char *argv[])
{
  try
    {
      using namespace Progetto;
      using namespace dealii;

      // Initialize MPI
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Parse command line arguments
      std::string parameter_file = "parameters.prm";
      if(argc > 1)
        parameter_file = argv[1];

      // Initialize ParameterHandler
      ParameterHandler prm;
      declare_parameters(prm);

      // Check if parameter file exists (only Rank 0 writes)
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          if(!std::filesystem::exists(parameter_file))
            {
              std::cout << "Parameter file '" << parameter_file
                        << "' not found.\n";
              std::cout << "Creating default parameter file...\n";
              std::ofstream out(parameter_file);
              prm.print_parameters(out, ParameterHandler::Text);
              out.close();
              std::cout << "Default parameter file created: " << parameter_file
                        << "\n";
              std::cout << "Please edit the file and run again.\n";
              return 0;
            }

          // Read parameter file
          std::cout << "Reading parameters from: " << parameter_file << "\n";
        }

      // Ensure all processes wait for file creation/reading before proceeding
      MPI_Barrier(MPI_COMM_WORLD);

      prm.parse_input(parameter_file);

      // --- FIX: Only Rank 0 clears the folder to avoid filesystem race
      // conditions ---
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          clear_solutions_folder();
        }

      // Ensure all processes wait until folder is cleared
      MPI_Barrier(MPI_COMM_WORLD);

      // --- Read Mesh Parameters ---
      prm.enter_subsection("Mesh");
      const bool mesh = prm.get_bool("generate_mesh");
      const int  cells_per_direction = prm.get_integer("cells_per_direction");
      prm.leave_subsection();

      // --- Read Physical Parameters ---
      prm.enter_subsection("Physical Parameters");
      const double       T_end = prm.get_double("final_time");
      const double       sigma = prm.get_double("source_width_sigma");
      const double       x0_x = prm.get_double("source_center_x");
      const double       x0_y = prm.get_double("source_center_y");
      const unsigned int N_val = prm.get_integer("source_frequency_N");
      const double       a_val = prm.get_double("oscillation_magnitude_A");
      prm.leave_subsection();

      // Validate critical parameters
      if(T_end <= 0.0)
        {
          std::cerr << "Error: Final time (T) must be strictly positive.\n";
          return 1;
        }
      if(sigma <= 0.0)
        {
          std::cerr
            << "Error: Source width (sigma) must be strictly positive.\n";
          return 1;
        }

      // --- Read Material Properties ---
      prm.enter_subsection("Material Properties");
      const double user_rho = prm.get_double("density");
      const double user_cp = prm.get_double("specific_heat");
      const double user_k = prm.get_double("thermal_conductivity");
      const double user_Q = prm.get_double("source_intensity");
      prm.leave_subsection();

      // Validate physical coefficients
      if(user_rho <= 0.0)
        {
          std::cerr << "Error: Density (rho) must be strictly positive.\n";
          return 1;
        }
      if(user_cp <= 0.0)
        {
          std::cerr
            << "Error: Specific Heat (c_p) must be strictly positive.\n";
          return 1;
        }
      if(user_k <= 0.0)
        {
          std::cerr
            << "Error: Thermal Conductivity (k) must be strictly positive.\n";
          return 1;
        }

      // --- Read Solver Settings ---
      prm.enter_subsection("Solver Settings");
      const double user_theta = prm.get_double("theta");
      const double user_tol = prm.get_double("time_step_tolerance");
      const double user_dt_min = prm.get_double("minimum_time_step");
      const bool   time_step_doubling = prm.get_bool("use_step_doubling");
      prm.leave_subsection();

      // Validate solver settings
      if(user_tol <= 0.0)
        {
          std::cerr
            << "Error: Time step tolerance must be strictly positive.\n";
          return 1;
        }
      if(user_dt_min <= 0.0)
        {
          std::cerr
            << "Error: Minimum time step (dt_min) must be strictly positive.\n";
          return 1;
        }

      // --- Read Simulation Control ---
      prm.enter_subsection("Simulation Control");
      unsigned int       run_mode = prm.get_integer("run_mode");
      bool               run_reference = prm.get_bool("run_reference");
      const double       dt0 = prm.get_double("initial_time_step");
      const unsigned int base_refine = prm.get_integer("base_refinement");
      const unsigned int pre_steps = prm.get_integer("pre_refinement_steps");
      const unsigned int refine_every = prm.get_integer("refine_every_n_steps");
      const bool         write_vtk = prm.get_bool("write_vtk");
      const bool   output_at_each = prm.get_bool("output_at_each_timestep");
      const double output_interval = prm.get_double("output_time_interval");
      prm.leave_subsection();

      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "\n============================================\n"
                    << "       SIMULATION CONFIGURATION             \n"
                    << "============================================\n"
                    << "[Mesh]\n"
                    << "  Generate mesh: " << (mesh ? "Yes" : "Load from file")
                    << "\n"
                    << "  Cells per direction: " << cells_per_direction << "\n"
                    << "\n[Physical Parameters]\n"
                    << "  Final time T: " << T_end << "\n"
                    << "  Source: sigma=" << sigma << ", (x0,y0)=(" << x0_x
                    << "," << x0_y << ")\n"
                    << "  Source: N=" << N_val << ", A=" << a_val << "\n"
                    << "\n[Material Properties]\n"
                    << "  Density (rho): " << user_rho << " kg/m^3\n"
                    << "  Specific heat (c_p): " << user_cp << " J/(kg K)\n"
                    << "  Thermal conductivity (k): " << user_k << " W/(m K)\n"
                    << "  Source intensity (Q): " << user_Q << " W/m^3\n"
                    << "\n[Solver Settings]\n"
                    << "  Theta: " << user_theta << " ("
                    << (user_theta == 0.0   ? "Explicit Euler"
                        : user_theta == 0.5 ? "Crank-Nicolson"
                        : user_theta == 1.0 ? "Implicit Euler"
                                            : "Mixed")
                    << ")\n"
                    << "  Time step tolerance: " << user_tol << "\n"
                    << "  Minimum time step: " << user_dt_min << "\n"
                    << "  Step-doubling: "
                    << (time_step_doubling ? "Yes" : "Heuristic") << "\n"
                    << "\n[Simulation Control]\n"
                    << "  Run mode: " << run_mode << " ("
                    << (run_mode == 0   ? "Full Comparison"
                        : run_mode == 1 ? "Fixed/Fixed"
                        : run_mode == 2 ? "Adaptive Space/Fixed Time"
                        : run_mode == 3 ? "Fixed Space/Adaptive Time"
                                        : "Adaptive/Adaptive")
                    << ")\n"
                    << "  Run reference: " << (run_reference ? "Yes" : "No")
                    << "\n"
                    << "  Initial time step: " << dt0 << "\n"
                    << "  Base refinement: " << base_refine << "\n"
                    << "  Pre-refinement steps: " << pre_steps << "\n"
                    << "  Refine every N steps: " << refine_every << "\n"
                    << "  Write VTK: " << (write_vtk ? "Yes" : "No") << "\n"
                    << "  Output mode: "
                    << (output_at_each ? "Each timestep (debug)"
                                       : "Fixed time intervals")
                    << "\n"
                    << "  Output time interval: " << output_interval << "\n"
                    << "============================================\n\n";
        }

      const std::string summary_path = "solutions/summary_comparison.csv";

      // Replicated Reference Strategy - Disabled for parallel execution (too
      // slow) Check if running in parallel
      const unsigned int mpi_size =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      if(run_reference && mpi_size > 1)
        {
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::cout << "\n"
                        << std::string(70, '=') << "\n"
                        << "WARNING: Reference solver disabled for MPI "
                           "execution (>1 process)\n"
                        << "Running the reference solution on every MPI rank "
                           "is prohibitively slow.\n"
                        << "L2 errors will be reported as NaN.\n"
                        << "To compute L2 errors, run with: mpirun -np 1 "
                           "./heat-equation <params>\n"
                        << std::string(70, '=') << "\n\n";
            }
          run_reference = false;
        }

      // Reference solver disabled for MPI
      std::unique_ptr<dealii::Functions::FEFieldFunction<2>> reference_function;
      std::unique_ptr<HeatEquation<2>>                       reference_solver;
      Vector<double> serial_reference_solution;

      if(run_reference)
        {
          // Run reference solver replicated on all ranks using MPI_COMM_SELF
          // This ensures every process has a copy for L2 error comparison
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "\n========== RUN REFERENCE (Replicated on all ranks "
                         "with MPI_COMM_SELF) ==========\n";

          int ref_cells = cells_per_direction;
          if(mesh)
            ref_cells = static_cast<int>(std::max(1, cells_per_direction * 2));

          HeatEquationParameters params;
          params.run_name = "reference";
          params.output_dir = "solutions/reference/";
          params.use_space_adaptivity = false;
          params.use_time_adaptivity = false;
          params.use_step_doubling = false;
          params.generate_mesh = mesh;
          params.cells_per_direction = ref_cells;
          params.source_frequency_N = N_val;
          params.source_width_sigma = sigma;
          params.source_magnitude_a = a_val;
          params.source_center_x = x0_x;
          params.source_center_y = x0_y;
          params.density = user_rho;
          params.specific_heat = user_cp;
          params.thermal_conductivity = user_k;
          params.source_intensity = user_Q;
          params.theta = user_theta;
          params.time_step_tolerance = user_tol;
          params.time_step_min = user_dt_min;
          params.end_time = T_end;
          params.initial_time_step = dt0 / 10.0;
          params.initial_global_refinement = base_refine + 2;
          params.n_adaptive_pre_refinement_steps = 0;
          params.refine_every_n_steps = refine_every;
          params.write_vtk = false;
          params.output_at_each_timestep = output_at_each;
          params.output_time_interval = output_interval;

          // Create reference solver with MPI_COMM_SELF so each rank solves it
          // independently
          reference_solver =
            std::make_unique<HeatEquation<2>>(params, MPI_COMM_SELF);
          reference_solver->run();

          // Extract solution to serial vector for FEFieldFunction
          serial_reference_solution.reinit(
            reference_solver->get_dof_handler().n_dofs());
          reference_solver->get_solution_serial(serial_reference_solution);

          // Create FEFieldFunction using the serial vector
          reference_function =
            std::make_unique<dealii::Functions::FEFieldFunction<2>>(
              reference_solver->get_dof_handler(), serial_reference_solution);
          reference_function->set_time(T_end);

          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "\nReference solution prepared for parallel comparison.\n";
        }
      else
        {
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "\nSkipping reference solver. L2 errors will be NaN.\n";
        }

      auto run_one = [&](const std::string &name, bool use_space, bool use_time,
                         bool use_sd) {
        HeatEquationParameters params;
        params.run_name = name;
        params.output_dir = "solutions/" + name + "/";
        params.use_space_adaptivity = use_space;
        params.use_time_adaptivity = use_time;
        params.use_step_doubling = use_sd;
        params.generate_mesh = mesh;
        params.cells_per_direction = cells_per_direction;
        params.source_frequency_N = N_val;
        params.source_width_sigma = sigma;
        params.source_magnitude_a = a_val;
        params.source_center_x = x0_x;
        params.source_center_y = x0_y;
        params.density = user_rho;
        params.specific_heat = user_cp;
        params.thermal_conductivity = user_k;
        params.source_intensity = user_Q;
        params.theta = user_theta;
        params.time_step_tolerance = user_tol;
        params.time_step_min = user_dt_min;
        params.end_time = T_end;
        params.initial_time_step = dt0;
        params.initial_global_refinement = base_refine;
        params.n_adaptive_pre_refinement_steps = pre_steps;
        params.refine_every_n_steps = refine_every;
        params.write_vtk = write_vtk;
        params.output_at_each_timestep = output_at_each;
        params.output_time_interval = output_interval;

        HeatEquation<2> solver(params);

        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "\n========== RUN " << name << " ==========\n";
        solver.run();

        double l2_err = std::numeric_limits<double>::quiet_NaN();
        if(reference_function)
          l2_err = solver.compute_L2_error_against(*reference_function);

        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::cout << "[" << name << "] L2 error at T = " << l2_err << "\n";
            append_comparison_csv(summary_path, name, l2_err,
                                  solver.get_stats());
          }
      };

      if(run_mode == 0)
        {
          run_one("fixed_space_fixed_time", false, false, false);
          run_one("adaptive_space_fixed_time", true, false, false);
          run_one("fixed_space_adaptive_time", false, true, time_step_doubling);
          run_one("adaptive_space_adaptive_time", true, true,
                  time_step_doubling);
          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::cout << "\nComparison finished. CSV saved to: "
                        << summary_path << "\n";
            }
        }
      else if(run_mode == 1)
        run_one("fixed_space_fixed_time", false, false, false);
      else if(run_mode == 2)
        run_one("adaptive_space_fixed_time", true, false, false);
      else if(run_mode == 3)
        run_one("fixed_space_adaptive_time", false, true, time_step_doubling);
      else if(run_mode == 4)
        run_one("adaptive_space_adaptive_time", true, true, time_step_doubling);

      return 0;
    }
  catch(std::exception &exc)
    {
      std::cerr << "\n\n----------------------------------------------------\n"
                << "Exception on processing:\n"
                << exc.what() << "\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch(...)
    {
      std::cerr << "\n\n----------------------------------------------------\n"
                << "Unknown exception!\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}