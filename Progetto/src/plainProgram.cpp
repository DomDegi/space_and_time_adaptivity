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

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <map>
#include <string>

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

namespace Progetto
{
  using namespace dealii;

  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation(bool space_adaptivity,
                 bool time_adaptivity,
                 bool step_doubling,
                 bool mesh,
                 int  cells_per_direction);

    void run();

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

    double       time;
    double       time_step;
    unsigned int timestep_number;

    bool use_space_adaptivity;

    bool use_time_adaptivity;
    bool use_step_doubling;

    double time_step_tolerance;
    double time_step_min;
    double time_step_max;
    double time_step_safety;

    const double theta;

    bool use_mesh;
    int  cells_per_direction;

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
  };

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
      , period(0.2)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;

  private:
    const double period;
  };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim>  &p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());

    const double time = this->get_time();

    const double a     = 5.0;
    const unsigned int N = 5;

    // sigma della sorgente (come nel tuo file originale)
    const double sigma = 0.5;

    Point<dim> x0;
    x0[0] = 0.5;
    x0[1] = 0.5;

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
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }

  template <int dim>
  HeatEquation<dim>::HeatEquation(bool space_adaptivity,
                                 bool time_adaptivity,
                                 bool step_doubling,
                                 bool mesh,
                                 int  cells_per_direction)
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 500.0)
    , timestep_number(0)
    , use_space_adaptivity(space_adaptivity)
    , use_time_adaptivity(time_adaptivity)
    , use_step_doubling(step_doubling)
    , time_step_tolerance(1e-8)
    , time_step_min(1e-6)
    , time_step_max(1e-1)
    , time_step_safety(0.9)
    , theta(0.5)
    , use_mesh(mesh)
    , cells_per_direction(cells_per_direction)
  {}

  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells() << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                    /*keep_constrained_dofs=*/true);
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
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations." << std::endl;
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

    RightHandSide<dim> rhs_function;

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
    std::filesystem::create_directories("solutions");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");
    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";

    std::ofstream output("solutions/" + filename);
    data_out.write_vtk(output);
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

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell : triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();

    for (const auto &cell : triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement();

    setup_system();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);

    // coerente per refinements DURANTE l'evoluzione
    old_solution.reinit(solution.size());
    old_solution = solution;
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement       = 2;
    const unsigned int n_adaptive_pre_refinement_steps = 4;

    // Mesh + setup
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

      setup_system();
    }

    // Imposto IC: u(t=0)=0
    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
    solution = old_solution;

    // ====== PRE-REFINEMENT COERENTE (senza goto) ======
    // Idea: faccio un "probe step" da t=0 a t=time_step, raffino sulla soluzione ottenuta,
    // poi resetto IC a zero e ripeto per n_adaptive_pre_refinement_steps volte.
    if (use_space_adaptivity)
    {
      for (unsigned int pre = 0; pre < n_adaptive_pre_refinement_steps; ++pre)
      {
        std::cout << "\n[Pre-refinement] step " << (pre + 1) << " / "
                  << n_adaptive_pre_refinement_steps << std::endl;

        const double dt_probe = time_step;
        Vector<double> u_probe(solution.size());

        // probe: da t=0 a t=dt_probe partendo da IC
        do_time_step(old_solution, dt_probe, dt_probe, u_probe);

        // ora 'solution' (e u_probe) contiene la soluzione a t=dt_probe -> uso per stimare errore
        refine_mesh(initial_global_refinement,
                    initial_global_refinement + n_adaptive_pre_refinement_steps);

        // IMPORTANTISSIMO: dopo il raffinamento, riparto coerentemente dalla IC (zero)
        VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), old_solution);
        solution = old_solution;
      }
    }

    // ====== INIZIO SIMULAZIONE VERA ======
    time            = 0.0;
    timestep_number = 0;

    // output iniziale sulla mesh finale (dopo pre-refinement)
    output_results();

    Vector<double> tmp(solution.size());
    Vector<double> forcing_terms(solution.size());

    const double end_time = 0.5;

    if (!use_time_adaptivity)
    {
      const double eps = 1e-12;

      while (time < end_time - eps)
      {
        const double dt = std::min(time_step, end_time - time);
        time += dt;
        ++timestep_number;

        std::cout << "Time step " << timestep_number
                  << " at t=" << time
                  << "  dt=" << dt << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1.0 - theta) * dt, tmp);

        RightHandSide<dim> rhs_function;

        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= dt * theta;

        rhs_function.set_time(time - dt);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms.add(dt * (1.0 - theta), tmp);

        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * dt, laplace_matrix);

        constraints.condense(system_matrix, system_rhs);

        solve_time_step();

        output_results();

        // Refinement DURANTE evoluzione: qui old_solution deve restare coerente col tempo corrente.
        if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
        {
          refine_mesh(initial_global_refinement,
                      initial_global_refinement + n_adaptive_pre_refinement_steps);

          tmp.reinit(solution.size());
          forcing_terms.reinit(solution.size());
        }

        old_solution = solution;
      }
    }
    else
    {
      const unsigned int p = 2;

      while (time < end_time)
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

            std::cout << "Time step " << timestep_number
                      << " accepted at t=" << time
                      << "  dt=" << dt
                      << "  error=" << error << std::endl;

            output_results();

            if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
            {
              refine_mesh(initial_global_refinement,
                          initial_global_refinement + n_adaptive_pre_refinement_steps);

              tmp.reinit(solution.size());
              forcing_terms.reinit(solution.size());
            }

            const double err_ratio =
              (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;

            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
            time_step = dt_new;
          }
          else
          {
            const double err_ratio =
              (error > 0.0) ? (time_step_tolerance * std::max(1.0, u_two.l2_norm()) / error) : 1e6;

            double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
            dt_new = std::max(dt_new, time_step_min);
            time_step = dt_new;

            std::cout << "Time step rejected at t=" << time
                      << "  dt_old=" << dt
                      << "  error=" << error
                      << "  new_dt=" << time_step << std::endl;
          }
        }
        else
        {
          const double t_new = time + dt;

          Vector<double> rhs_new(solution.size());
          RightHandSide<dim> rhs_function;

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

          mass_matrix.vmult(system_rhs, old_solution);

          laplace_matrix.vmult(tmp, old_solution);
          system_rhs.add(-(1.0 - theta) * dt, tmp);

          rhs_function.set_time(t_new);
          VectorTools::create_right_hand_side(dof_handler,
                                              QGauss<dim>(fe.degree + 1),
                                              rhs_function,
                                              tmp);
          forcing_terms = tmp;
          forcing_terms *= dt * theta;

          rhs_function.set_time(time);
          VectorTools::create_right_hand_side(dof_handler,
                                              QGauss<dim>(fe.degree + 1),
                                              rhs_function,
                                              tmp);
          forcing_terms.add(dt * (1.0 - theta), tmp);

          system_rhs += forcing_terms;

          system_matrix.copy_from(mass_matrix);
          system_matrix.add(theta * dt, laplace_matrix);
          constraints.condense(system_matrix, system_rhs);

          solve_time_step();

          time = t_new;
          ++timestep_number;
          old_solution = solution;

          std::cout << "Heuristic step " << timestep_number
                    << " at t=" << time
                    << "  dt=" << dt
                    << "  err_est=" << error_est << std::endl;

          output_results();

          if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement + n_adaptive_pre_refinement_steps);

            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
          }

          double factor = 1.0;
          if (error_est > 2.0 * tol)
            factor = 0.5;
          else if (error_est < 0.25 * tol)
            factor = 2.0;

          double dt_new = dt * factor * time_step_safety;
          dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
          time_step = dt_new;
        }
      }
    }
  }

} // namespace Progetto

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
        std::cerr << "Numero di celle non valido. Deve essere un intero positivo." << std::endl;
        return 1;
      }
      std::cout << "Generazione di una mesh con " << cells_per_direction
                << " celle per direzione." << std::endl;
    }

    const bool space = ask_bool("Vuoi abilitare l'adattività spaziale?");
    const bool time  = ask_bool("Vuoi abilitare l'adattività temporale?");

    if (time)
    {
      const bool type = ask_bool("Se sì, vuoi usare lo step-doubling? (altrimenti verrà usata l'euristica economica)");
      HeatEquation<2> heat_equation_solver(space, time, type, mesh, cells_per_direction);
      heat_equation_solver.run();
    }
    else
    {
      HeatEquation<2> heat_equation_solver(space, time, false, mesh, cells_per_direction);
      heat_equation_solver.run();
    }
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

  return 0;
}
