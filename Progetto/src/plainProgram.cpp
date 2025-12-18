#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <sstream>
#include <chrono>
#include <limits>
#include <memory>

void clear_solutions_folder()
{
  std::filesystem::create_directories("solutions");
  for (const auto &entry : std::filesystem::directory_iterator("solutions"))
    std::filesystem::remove_all(entry.path());
}

bool ask_bool(const std::string &question)
{
  while (true)
  {
    std::cout << question << " (s/n): ";
    std::string input;
    std::cin >> input;

    std::transform(input.begin(), input.end(), input.begin(), ::tolower);

    if (input == "s" || input == "si" || input == "y" || input == "yes" || input == "1")
      return true;

    if (input == "n" || input == "no" || input == "0")
      return false;

    std::cout << "Input non valido. Riprova.\n";
  }
}

double ask_double_default(const std::string &question, const double default_value)
{
  while (true)
  {
    std::cout << question << " [default=" << default_value << "]: ";
    std::string line;
    std::getline(std::cin >> std::ws, line);

    if (line.empty())
      return default_value;

    std::stringstream ss(line);
    double v;
    if (ss >> v)
      return v;

    std::cout << "Input non valido. Riprova.\n";
  }
}

unsigned int ask_uint_default(const std::string &question, const unsigned int default_value)
{
  while (true)
  {
    std::cout << question << " [default=" << default_value << "]: ";
    std::string line;
    std::getline(std::cin >> std::ws, line);

    if (line.empty())
      return default_value;

    std::stringstream ss(line);
    int v;
    if ((ss >> v) && v > 0)
      return static_cast<unsigned int>(v);

    std::cout << "Input non valido. Inserisci un intero positivo.\n";
  }
}

namespace Progetto
{
  using namespace dealii;

  template <int dim>
  class HeatEquation
  {
  public:
    struct RunStats
    {
      double cpu_seconds_total = 0.0;

      unsigned int n_linear_solves = 0;
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
      unsigned int dof_min = std::numeric_limits<unsigned int>::max();
      unsigned int dof_max = 0;
      unsigned int dof_samples = 0;

      unsigned int cells_min = std::numeric_limits<unsigned int>::max();
      unsigned int cells_max = 0;

      void reset() { *this = RunStats(); }

      void sample_dofs_and_cells(const unsigned int ndofs, const unsigned int ncells)
      {
        dof_min = std::min(dof_min, ndofs);
        dof_max = std::max(dof_max, ndofs);
        dof_sum += ndofs;
        dof_samples++;

        cells_min = std::min(cells_min, ncells);
        cells_max = std::max(cells_max, ncells);
      }

      void register_cg_iterations(const unsigned int it)
      {
        cg_iterations_sum += it;
        cg_iterations_min = std::min(cg_iterations_min, it);
        cg_iterations_max = std::max(cg_iterations_max, it);
      }

      void register_time_step_attempt(const double dt, const bool accepted)
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

      double dt_mean() const
      {
        if (n_time_steps_accepted == 0) return 0.0;
        return dt_sum / static_cast<double>(n_time_steps_accepted);
      }

      double dof_mean() const
      {
        if (dof_samples == 0) return 0.0;
        return static_cast<double>(dof_sum) / static_cast<double>(dof_samples);
      }
    };

    HeatEquation(const std::string &run_name_in,
                 const std::string &output_dir_in,
                 bool space_adaptivity,
                 bool time_adaptivity,
                 bool step_doubling,
                 bool mesh,
                 int  cells_per_direction,
                 unsigned int rhs_N_in,
                 double rhs_sigma_in,
                 double rhs_x0_x_in,
                 double rhs_x0_y_in,
                 double end_time_in,
                 double initial_dt_in,
                 unsigned int initial_global_refinement_in,
                 unsigned int n_adaptive_pre_refinement_steps_in,
                 unsigned int refine_every_n_steps_in,
                 bool write_vtk_in);

    void run();

    const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }
    const Vector<double>  &get_solution()   const { return solution; }
    const Triangulation<dim> &get_triangulation() const { return triangulation; }
    const FE_Q<dim> &get_fe() const { return fe; }

    const RunStats &get_stats() const { return stats; }

    double compute_L2_error_against(const Function<dim> &reference_function) const;

  private:
    void setup_system();
    void solve_time_step();

    void do_time_step(const Vector<double> &u_old,
                      const double          dt,
                      const double          t_new,
                      Vector<double> &      u_out);

    void output_results() const;

    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    void log_mesh_event(const unsigned int marked_refine,
                        const unsigned int marked_coarsen) const;

    void log_time_event(const double t_now,
                        const double dt_now,
                        const int accepted,
                        const double err_est,
                        const double new_dt) const;

    Triangulation<dim> triangulation;
    const FE_Q<dim>    fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    double       time = 0.0;
    double       time_step = 1. / 500.0;
    unsigned int timestep_number = 0;

    bool use_space_adaptivity = false;

    bool use_time_adaptivity = false;
    bool use_step_doubling   = false;

    double time_step_tolerance = 1e-8;
    double time_step_min       = 1e-6;
    double time_step_max       = 1e-1;
    double time_step_safety    = 0.9;

    const double theta;

    bool use_mesh = true;
    int  cells_per_direction = 0;

    unsigned int rhs_N = 5;
    double rhs_sigma  = 0.5;
    Point<dim> rhs_x0;

    double end_time = 0.5;

    unsigned int initial_global_refinement = 2;
    unsigned int n_adaptive_pre_refinement_steps = 4;
    unsigned int refine_every_n_steps = 5;

    bool write_vtk = true;

    std::string run_name;
    std::string output_dir;

    RunStats stats;

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
  };

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const unsigned int N_in,
                  const double sigma_in,
                  const Point<dim> &x0_in)
      : Function<dim>()
      , N(N_in)
      , sigma(sigma_in)
      , x0(x0_in)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;

  private:
    const unsigned int N;
    const double sigma;
    const Point<dim> x0;
  };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim>  &p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    const double time = this->get_time();
    const double a = 5.0;

    const double g =
      std::exp(-a * std::cos(2.0 * N * numbers::PI * time)) / std::exp(a);

    double r2 = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
    {
      const double diff = p[d] - x0[d];
      r2 += diff * diff;
    }

    const double h = std::exp(-r2 / (sigma * sigma));
    return g * h;
  }

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
                                 double rhs_x0_x_in,
                                 double rhs_x0_y_in,
                                 double end_time_in,
                                 double initial_dt_in,
                                 unsigned int initial_global_refinement_in,
                                 unsigned int n_adaptive_pre_refinement_steps_in,
                                 unsigned int refine_every_n_steps_in,
                                 bool write_vtk_in)
    : fe(1)
    , dof_handler(triangulation)
    , theta(0.5)
  {
    run_name = run_name_in;
    output_dir = output_dir_in;

    use_space_adaptivity = space_adaptivity;
    use_time_adaptivity  = time_adaptivity;
    use_step_doubling    = step_doubling;

    use_mesh = mesh;
    cells_per_direction = cells_per_direction_in;

    rhs_N = rhs_N_in;
    rhs_sigma = rhs_sigma_in;
    rhs_x0[0] = rhs_x0_x_in;
    rhs_x0[1] = rhs_x0_y_in;

    end_time = end_time_in;

    time_step = initial_dt_in;

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

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);

    const double d = 1.0;
    const double c = 1.0;
    mass_matrix *= c * d;

    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    const double k = 1.0;
    laplace_matrix *= k;

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    const double rhs_norm = system_rhs.l2_norm();
    const double tol = std::max(1e-14, 1e-8 * rhs_norm);

    SolverControl            solver_control(1000, tol);
    SolverCG<Vector<double>> cg(solver_control);

    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    const unsigned int its = solver_control.last_step();
    stats.n_linear_solves++;
    stats.register_cg_iterations(its);

    std::cout << "     " << its << " CG iterations.\n";
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

    RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_x0);

    rhs_function.set_time(t_new);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree + 1),
                                        rhs_function,
                                        tmp);
    forcing_terms_local = tmp;
    forcing_terms_local *= dt * theta;

    rhs_function.set_time(t_new - dt);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree + 1),
                                        rhs_function,
                                        tmp);
    forcing_terms_local.add(dt * (1.0 - theta), tmp);

    system_rhs += forcing_terms_local;

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * dt, laplace_matrix);

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

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";

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

    log << timestep_number << ","
        << time << ","
        << triangulation.n_active_cells() << ","
        << dof_handler.n_dofs() << ","
        << triangulation.n_levels() << ","
        << marked_refine << ","
        << marked_coarsen << "\n";
  }

  template <int dim>
  void HeatEquation<dim>::log_time_event(const double t_now,
                                        const double dt_now,
                                        const int accepted,
                                        const double err_est,
                                        const double new_dt) const
  {
    std::filesystem::create_directories(output_dir);
    const std::string log_path = output_dir + "/time_log.csv";
    const bool write_header = !std::filesystem::exists(log_path);

    std::ofstream log(log_path, std::ios::app);
    if (write_header)
      log << "timestep,time,dt,accepted,error_est,new_dt\n";

    log << timestep_number << ","
        << t_now << ","
        << dt_now << ","
        << accepted << ","
        << err_est << ","
        << new_dt << "\n";
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6,
                                                      0.4);

    if (triangulation.n_levels() > 0)
    {
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() >= static_cast<int>(max_grid_level))
          cell->clear_refine_flag();

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->level() <= static_cast<int>(min_grid_level))
          cell->clear_coarsen_flag();
    }

    unsigned int marked_refine  = 0;
    unsigned int marked_coarsen = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->refine_flag_set())
        ++marked_refine;
      if (cell->coarsen_flag_set())
        ++marked_coarsen;
    }

    std::cout << "[" << run_name << "][AMR] refine_mesh at t=" << time
              << " (step " << timestep_number << ")"
              << " | active_cells=" << triangulation.n_active_cells()
              << " dofs=" << dof_handler.n_dofs()
              << " levels=" << triangulation.n_levels()
              << " | marked_refine=" << marked_refine
              << " marked_coarsen=" << marked_coarsen
              << "\n";

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

        refine_mesh(initial_global_refinement,
                    initial_global_refinement + n_adaptive_pre_refinement_steps);

        VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
        solution = old_solution;

        stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
      }
    }

    time            = 0.0;
    timestep_number = 0;

    if (write_vtk)
      output_results();

    const double eps = 1e-12;

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

        if (write_vtk)
          output_results();

        if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
        {
          refine_mesh(initial_global_refinement,
                      initial_global_refinement + n_adaptive_pre_refinement_steps);
          stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
        }
      }
    }
    else
    {
      const unsigned int p = 2;

      while (time < end_time - eps)
      {
        double dt = std::min(time_step, end_time - time);

        if (use_step_doubling)
        {
          Vector<double> u_one(solution.size());
          Vector<double> u_two(solution.size());
          const double t_one = time + dt;

          do_time_step(old_solution, dt, t_one, u_one);

          const double dt2 = 0.5 * dt;
          Vector<double> u_half(solution.size());
          do_time_step(old_solution, dt2, time + dt2, u_half);
          do_time_step(u_half, dt2, t_one, u_two);

          Vector<double> diff(u_two);
          diff -= u_one;
          const double error = diff.l2_norm();

          const double sol_norm   = std::max(1.0, u_two.l2_norm());
          const double tol_scaled = time_step_tolerance * sol_norm + 1e-16;

          if (error <= tol_scaled)
          {
            time = t_one;
            ++timestep_number;
            solution = u_two;
            old_solution = solution;

            std::cout << "[" << run_name << "] Time step " << timestep_number
                      << " accepted at t=" << time
                      << "  dt=" << dt
                      << "  error=" << error << "\n";

            stats.register_time_step_attempt(dt, true);
            stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

            if (write_vtk)
              output_results();

            if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
            {
              refine_mesh(initial_global_refinement,
                          initial_global_refinement + n_adaptive_pre_refinement_steps);
              stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
            }

            const double err_ratio =
              (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;

            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
            time_step = dt_new;

            log_time_event(time, dt, 1, error, time_step);
          }
          else
          {
            const double err_ratio =
              (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;

            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::max(dt_new, time_step_min);
            time_step = dt_new;

            std::cout << "[" << run_name << "] Time step rejected at t=" << time
                      << "  dt_old=" << dt
                      << "  error=" << error
                      << "  new_dt=" << time_step << "\n";

            stats.register_time_step_attempt(dt, false);
            log_time_event(time, dt, 0, error, time_step);
          }
        }
        else
        {
          const double t_new = time + dt;

          Vector<double> rhs_new(solution.size());
          RightHandSide<dim> rhs_function(rhs_N, rhs_sigma, rhs_x0);

          rhs_function.set_time(t_new);
          VectorTools::create_right_hand_side(dof_handler,
                                              QGauss<dim>(fe.degree + 1),
                                              rhs_function,
                                              rhs_new);

          Vector<double> rhs_old(solution.size());
          rhs_function.set_time(time);
          VectorTools::create_right_hand_side(dof_handler,
                                              QGauss<dim>(fe.degree + 1),
                                              rhs_function,
                                              rhs_old);

          Vector<double> rhs_diff = rhs_new;
          rhs_diff -= rhs_old;

          const double rhs_diff_norm = rhs_diff.l2_norm();
          const double rhs_norm      = std::max(1.0, rhs_new.l2_norm());

          const double error_est = dt * rhs_diff_norm / rhs_norm;
          const double tol       = time_step_tolerance;

          Vector<double> u_new(solution.size());
          do_time_step(old_solution, dt, t_new, u_new);

          time = t_new;
          ++timestep_number;
          solution = u_new;
          old_solution = solution;

          std::cout << "[" << run_name << "] Heuristic step " << timestep_number
                    << " at t=" << time
                    << "  dt=" << dt
                    << "  err_est=" << error_est << "\n";

          stats.register_time_step_attempt(dt, true);
          stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());

          if (write_vtk)
            output_results();

          if ((timestep_number % refine_every_n_steps == 0) && use_space_adaptivity)
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement + n_adaptive_pre_refinement_steps);
            stats.sample_dofs_and_cells(dof_handler.n_dofs(), triangulation.n_active_cells());
          }

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
      }
    }

    const auto t_end = std::chrono::steady_clock::now();
    stats.cpu_seconds_total =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

    std::cout << "[" << run_name << "] DONE. CPU seconds = " << stats.cpu_seconds_total << "\n";
  }

}

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

  const unsigned int cg_min = (st.cg_iterations_min == std::numeric_limits<unsigned int>::max())
                                ? 0u
                                : st.cg_iterations_min;

  const unsigned int dof_min = (st.dof_min == std::numeric_limits<unsigned int>::max())
                                 ? 0u
                                 : st.dof_min;

  const unsigned int cells_min = (st.cells_min == std::numeric_limits<unsigned int>::max())
                                   ? 0u
                                   : st.cells_min;

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

    const bool mesh = ask_bool("Vuoi generare una mesh o importarla da file?");
    int cells_per_direction = 0;

    if (mesh)
    {
      std::cout << "Inserisci il numero di celle per direzione (intero positivo): ";
      std::cin >> cells_per_direction;
      if (cells_per_direction <= 0)
      {
        std::cerr << "Numero di celle non valido. Deve essere un intero positivo.\n";
        return 1;
      }
      std::cout << "Generazione di una mesh con " << cells_per_direction << " celle per direzione.\n";
    }

    const double T_end       = ask_double_default("Inserisci T (tempo finale)", 0.5);
    const double sigma       = ask_double_default("Inserisci sigma (delta) della sorgente", 0.5);
    const double x0_x        = ask_double_default("Inserisci x0_x (centro sorgente)", 0.5);
    const double x0_y        = ask_double_default("Inserisci x0_y (centro sorgente)", 0.5);
    const unsigned int N_val = ask_uint_default("Inserisci N (frequenza temporale g(t))", 5);

    const double dt0 = 1.0 / 500.0;
    const unsigned int base_refine = 2;
    const unsigned int pre_steps   = 4;
    const unsigned int refine_every = 5;

    const bool do_comparison = true;
    const bool write_vtk = true;

    const double dt_ref = dt0 / 10.0;
    const unsigned int ref_refine = base_refine + 2;

    const bool time_step_doubling =
      ask_bool("Time adaptivity: vuoi usare step-doubling? (altrimenti euristica economica)");

    std::unique_ptr<HeatEquation<2>> reference_solver;
    std::unique_ptr<dealii::Functions::FEFieldFunction<2>> reference_function;

    if (do_comparison)
    {
      int ref_cells = cells_per_direction;
      if (mesh)
        ref_cells = static_cast<int>(std::max(1, cells_per_direction * 2));

      reference_solver = std::make_unique<HeatEquation<2>>(
        "reference",
        "solutions/reference",
        false,
        false,
        false,
        mesh,
        ref_cells,
        N_val, sigma, x0_x, x0_y, T_end,
        dt_ref,
        ref_refine,
        0,
        refine_every,
        false);

      std::cout << "\n========== RUN REFERENCE (fixed mesh + fixed dt) ==========\n";
      reference_solver->run();

      reference_function = std::make_unique<dealii::Functions::FEFieldFunction<2>>(
        reference_solver->get_dof_handler(),
        reference_solver->get_solution());
      reference_function->set_time(T_end);
    }

    if (do_comparison)
    {
      const std::string summary_path = "solutions/summary_comparison.csv";

      auto run_one = [&](const std::string &name,
                         const bool use_space,
                         const bool use_time,
                         const bool use_sd)
      {
        std::string outdir = "solutions/" + name;

        HeatEquation<2> solver(
          name,
          outdir,
          use_space,
          use_time,
          use_sd,
          mesh,
          cells_per_direction,
          N_val, sigma, x0_x, x0_y, T_end,
          dt0,
          base_refine,
          pre_steps,
          refine_every,
          write_vtk);

        std::cout << "\n========== RUN " << name << " ==========\n";
        solver.run();

        double l2_err = std::numeric_limits<double>::quiet_NaN();
        if (reference_function)
          l2_err = solver.compute_L2_error_against(*reference_function);

        std::cout << "[" << name << "] L2 error at T = " << l2_err << "\n";

        append_comparison_csv(summary_path, name, l2_err, solver.get_stats());
      };

      run_one("fixed_space_fixed_time", false, false, false);
      run_one("adaptive_space_fixed_time", true,  false, false);
      run_one("fixed_space_adaptive_time", false, true,  time_step_doubling);
      run_one("adaptive_space_adaptive_time", true,  true,  time_step_doubling);

      std::cout << "\nConfronto completato. CSV: " << summary_path << "\n";
      return 0;
    }

    return 0;
  }
  catch (std::exception &exc)
  {
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Exception on processing:\n" << exc.what() << "\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "\n\n----------------------------------------------------\n";
    std::cerr << "Unknown exception!\nAborting!\n";
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }
}
