 
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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
#include <deal.II/grid/grid_in.h>

 
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>

void clear_solutions_folder()
{
    for (const auto &entry : std::filesystem::directory_iterator("solutions"))
    {
        std::filesystem::remove_all(entry.path());
    }
}

bool ask_bool(const std::string &question)
{
    while (true)
    {
        std::cout << question << " (s/n): ";
        std::string input;
        std::cin >> input;

        // converto in lowercase
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
    HeatEquation(bool space_adaptivity, bool time_adaptivity, bool step_doubling, bool mesh, int cells_per_direction);
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

    //flag per attività spaziale
    bool use_space_adaptivity; // abilita/disabilita adattività spaziale

    // flags per adattività temporale
    bool use_time_adaptivity; // abilita/disabilita adattività temporale
    bool use_step_doubling;   // true -> step-doubling, false -> heuristica economica

    // parametri per adattività temporale
    double time_step_tolerance;
    double time_step_min;
    double time_step_max;
    double time_step_safety;
 
    const double theta;

    // parametri per la mesh
    bool use_mesh; // true -> genera mesh, false -> importa da file
    int cells_per_direction; // numero di celle per direzione (se genera mesh)
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
    AssertIndexRange(component, 1); //Verifica che component sia < 1
    Assert(dim == 2, ExcNotImplemented());
 
    const double time = this->get_time();
    /*
    const double point_within_period =
      (time / period - std::floor(time / period));
    
    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
    */
    

    const double a     = 5.0;
    const unsigned int N = 5;
    const double sigma = 1.0;           // <-- scegli tu il valore di σ
    Point<dim> x0;
    x0[0] = 0;
    x0[1] = 0;
    
    // g(t) = exp(-a cos(2 N pi t)) / exp(a)
    const double g = std::exp(-a * std::cos(2.0 * N * numbers::PI * time))
                    / std::exp(a);

    // |x - x0|^2
    double r2 = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const double diff = p[d] - x0[d];
        r2 += diff * diff;
      }

    // h(x) = exp( - |x - x0|^2 / sigma^2 )
    const double h = std::exp(-r2 / (sigma * sigma));

    // f(x,t) = g(t) * h(x)
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
                                 bool step_doubling, bool mesh, int cells_per_direction)
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 500)
    , timestep_number(0)
    , use_space_adaptivity(space_adaptivity) //Se scegliamo true utilizziamo l'adattività spaziale
    , use_time_adaptivity(time_adaptivity) //Se scegliamo true utilizziamo la time adaptivity
    , use_step_doubling(step_doubling) //Se scegliamo false utilizziamo la heuristica economica
    , time_step_tolerance(1e-4)
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
                << "Number of active cells: " << triangulation.n_active_cells()
                << std::endl
                << "Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl
                << std::endl;
  
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
  
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      /*keep_constrained_dofs = */ true);
      sparsity_pattern.copy_from(dsp);
  
      mass_matrix.reinit(sparsity_pattern);
      laplace_matrix.reinit(sparsity_pattern);
      system_matrix.reinit(sparsity_pattern);
  
      MatrixCreator::create_mass_matrix(dof_handler,
                                        QGauss<dim>(fe.degree + 1),
                                        mass_matrix);
      MatrixCreator::create_laplace_matrix(dof_handler,
                                          QGauss<dim>(fe.degree + 1),
                                          laplace_matrix);
  
      solution.reinit(dof_handler.n_dofs());
      old_solution.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());
  }

  
 
  // Risolve il sistema lineare per un singolo passo temporale con CG
  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
 
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
 
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
 
    constraints.distribute(solution); // Assicura che i contraints siano rispettati pulendo la soluzione
 
    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }


  // Esegue un singolo passo temporale di lunghezza dt, partendo da u_old
  // e considerando che il tempo al nuovo passo è t_new (== t_old + dt).
  // Restituisce la soluzione in u_out.
  template <int dim>
  void HeatEquation<dim>::do_time_step(const Vector<double> &u_old,
                                       const double          dt,
                                       const double          t_new,
                                       Vector<double> &      u_out)
  {
    Vector<double> tmp(u_old.size());
    Vector<double> forcing_terms_local(u_old.size());

    // system_rhs = M * u_old
    mass_matrix.vmult(system_rhs, u_old);

    // tmp = K * u_old
    laplace_matrix.vmult(tmp, u_old);
    system_rhs.add(-(1 - theta) * dt, tmp);

    // contributi sorgente: f(t_new) e f(t_new - dt)
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

    forcing_terms_local.add(dt * (1 - theta), tmp);

    system_rhs += forcing_terms_local;

    // system_matrix = M + theta*dt*K
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * dt, laplace_matrix);

    constraints.condense(system_matrix, system_rhs);

    // applica BC
    BoundaryValues<dim> boundary_values_function;
    boundary_values_function.set_time(t_new);
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_values_function,
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

    // risolvi sistema (popola `solution` membro)
    solve_time_step();

    // copia risultato in u_out
    u_out = solution;
  }
 
 
 
  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;
 
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");
 
    data_out.build_patches();
 
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
 
    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
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
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
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


  }
 
 
 
  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement       = 2;
    const unsigned int n_adaptive_pre_refinement_steps = 4;
    
    if (use_mesh)
    {
        GridGenerator::subdivided_hyper_cube(triangulation, cells_per_direction); //qui viene definita la geometria del mesh a forma di L
        triangulation.refine_global(initial_global_refinement); //refinamento globale iniziale
        setup_system(); //setup del sistema
    }else
    {
      const std::string mesh_file_name = "../mesh/mesh-input.msh";

      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);

      std::ifstream mesh_file(mesh_file_name);
      AssertThrow(mesh_file, ExcMessage("Could not open mesh file"));

      grid_in.read_msh(mesh_file);   // ⚠️ deve essere QUAD o HEX mesh --> TO DO permettere anche TRI o TET

      setup_system();
    }               
    
     
 
    unsigned int pre_refinement_step = 0;
 
    Vector<double> tmp;
    Vector<double> forcing_terms;
 
  start_time_iteration:
 
    time            = 0.0;
    timestep_number = 0;
 
    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());
 
 
    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<dim>(),
                             old_solution);
    solution = old_solution;
 
    output_results();
 
    const double end_time = 0.5; // Tempo finale

    /*******************************************************************************************************+****
     * 
     * La matemaica dietro la risoluzione dell'equazione del calore con il metodo di Crank-Nicolson 
     * 
     * La PDE in forma semi-discreta (FEM nello spazio, tempo ancora continuo) è: M u_puntato(t) + K u(t) = F(t)
     * dove M è la matrice di massa, K è la matrice di Laplace, F(t) è il vettore dei forcing(right-hand side)
     * 
     * Discretizzando temporalmente con Crank-Nicolson (θ=0.5) tra t^(n-1) e t^(n-1)+dt si ottiene:
     *    (M + theta*dt*K) u^n =  (M - (1-theta)*dt*K) u^{n-1}+ dt*[ (1-theta)*F^{n-1} + theta*F^n ]
     * 
     * Tramite la sezione 1 del codice si calcola il primo termine di rhs:
     *    (M - (1-theta)*dt*K) u^{n-1}
     * 
     * Tramite la sezione 2 del codice si calcolano i termini di forcing:
     *    dt*[ (1-theta)*F^{n-1} + theta*F^n ]
     * 
     * Tramite la sezione 3 del codice si costruisce la matrice di sistema:
     *    (M + theta*dt*K) u^n e viene posta uguale al rhs calcolato precedentemente
     * 
     * Tramite la sezione 4 del codice si applicano le condizioni al contorno
     * 
     * Nella sezione 5 del codice si risolve il sistema lineare per ottenere u^n e si scrive l'output
     * 
     ***********************************************************************************************************/

    if (!use_time_adaptivity)
      {
        // comportamento originale: passo fisso
        while (time <= end_time)
          {
            time += time_step;
            ++timestep_number;

            std::cout << "Time step " << timestep_number << " at t=" << time
                      << std::endl;

            /***************************************        (1)        ***************************************/

            mass_matrix.vmult(system_rhs, old_solution);

            laplace_matrix.vmult(tmp, old_solution);
            system_rhs.add(-(1 - theta) * time_step, tmp);
            
            /***************************************        (2)        ***************************************/

            RightHandSide<dim> rhs_function;
            rhs_function.set_time(time);
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree + 1),
                                                rhs_function,
                                                tmp);
            forcing_terms = tmp;
            forcing_terms *= time_step * theta;

            rhs_function.set_time(time - time_step);
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree + 1),
                                                rhs_function,
                                                tmp);

            forcing_terms.add(time_step * (1 - theta), tmp);

            system_rhs += forcing_terms;

            /***************************************        (3)        ***************************************/

            system_matrix.copy_from(mass_matrix);
            system_matrix.add(theta * time_step, laplace_matrix);

            constraints.condense(system_matrix, system_rhs);

            /***************************************        (4)        ***************************************/

            {
              BoundaryValues<dim> boundary_values_function;
              boundary_values_function.set_time(time);

              std::map<types::global_dof_index, double> boundary_values;
              VectorTools::interpolate_boundary_values(dof_handler,
                                                       0,
                                                       boundary_values_function,
                                                       boundary_values);

              MatrixTools::apply_boundary_values(boundary_values,
                                                 system_matrix,
                                                 solution,
                                                 system_rhs);
            }

            /***************************************        (5)        ***************************************/

            solve_time_step();

            output_results();

            /*************************************************************************************************/

            if ((timestep_number == 1) &&
                (pre_refinement_step < n_adaptive_pre_refinement_steps) && use_space_adaptivity)
              {
                refine_mesh(initial_global_refinement,
                            initial_global_refinement +
                              n_adaptive_pre_refinement_steps);
                ++pre_refinement_step;

                std::cout << std::endl;

                goto start_time_iteration;
              }
            else if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
              {
                refine_mesh(initial_global_refinement,
                            initial_global_refinement +
                              n_adaptive_pre_refinement_steps);
                tmp.reinit(solution.size());
                forcing_terms.reinit(solution.size());
              }

            old_solution = solution;
          }
      }
    else
      {
        // adattività temporale: scegli tra step-doubling o heuristica
        const unsigned int p = 2; // ordine approssimativo per Crank-Nicolson

        while (time < end_time)
          {
            double dt = std::min(time_step, end_time - time);

            if (use_step_doubling)
              {
                // step-doubling: soluzione con dt e con due mezzi-passi dt/2
                Vector<double> u_one(solution.size());
                Vector<double> u_two(solution.size());

                const double t_one = time + dt;

                do_time_step(old_solution, dt, t_one, u_one);

                const double dt2 = dt * 0.5;
                Vector<double> u_half(solution.size());
                //Per calcolare l'errore faccio due mezzi-passi, ovvero:
                do_time_step(old_solution, dt2, time + dt2, u_half);
                do_time_step(u_half, dt2, t_one, u_two);
                //Infatti devo confrontare u_one con u_two, ma devono essere due soluzioni allo stesso
                //istante di tempo t_one = time + dt

                Vector<double> diff(u_two); //calcolo differenza tra le soluzioni a dt e dt/2
                diff -= u_one;
                const double error = diff.l2_norm();

                const double sol_norm = std::max(1.0, u_two.l2_norm());
                const double tol_scaled = time_step_tolerance * sol_norm + 1e-16;

                if (error <= tol_scaled)
                  {
                    // accetta
                    time = t_one;
                    ++timestep_number;
                    solution = u_two;
                    old_solution = solution;
                    std::cout << "Time step " << timestep_number << " accepted at t=" << time
                              << "  dt=" << dt << "  error=" << error << std::endl;

                    output_results();

                    if ((timestep_number == 1) &&
                        (pre_refinement_step < n_adaptive_pre_refinement_steps) && use_space_adaptivity)
                      {
                        refine_mesh(initial_global_refinement,
                                    initial_global_refinement +
                                      n_adaptive_pre_refinement_steps);
                        ++pre_refinement_step;

                        std::cout << std::endl;

                        goto start_time_iteration;
                      }
                    else if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
                      {
                        refine_mesh(initial_global_refinement,
                                    initial_global_refinement +
                                      n_adaptive_pre_refinement_steps);
                        tmp.reinit(solution.size());
                        forcing_terms.reinit(solution.size());
                      }

                    const double err_ratio = (error > 0.0) ? (time_step_tolerance * sol_norm / error) : 1e6;
                    double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
                    dt_new = std::min(std::max(dt_new, time_step_min), time_step_max);
                    time_step = dt_new;
                  }
                else
                  {
                    // rifiuta e riduci dt
                    const double err_ratio = (error > 0.0) ? (time_step_tolerance * std::max(1.0, u_two.l2_norm()) / error) : 1e6;
                    double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
                    dt_new = std::max(dt_new, time_step_min);
                    time_step = dt_new;
                    std::cout << "Time step rejected at t=" << time << "  dt_old=" << dt
                              << "  error=" << error << "  new_dt=" << time_step << std::endl;
                  }
              }
            else
              {
                // heuristica economica: stima della variazione del RHS
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

                // stima d'errore proporzionale a dt * ||rhs_diff||
                const double error_est = dt * rhs_diff_norm;
                const double tol_scaled = time_step_tolerance * std::max(1.0, old_solution.l2_norm()) + 1e-16;

                // costruiamo e risolviamo il sistema con dt (come prima)
                mass_matrix.vmult(system_rhs, old_solution);
                laplace_matrix.vmult(tmp, old_solution);
                system_rhs.add(-(1 - theta) * dt, tmp);

                // forcing terms at t_new and t_old
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
                forcing_terms.add(dt * (1 - theta), tmp);

                system_rhs += forcing_terms;
                system_matrix.copy_from(mass_matrix);
                system_matrix.add(theta * dt, laplace_matrix);
                constraints.condense(system_matrix, system_rhs);

                BoundaryValues<dim> boundary_values_function;
                boundary_values_function.set_time(t_new);
                std::map<types::global_dof_index, double> boundary_values;
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         boundary_values_function,
                                                         boundary_values);
                MatrixTools::apply_boundary_values(boundary_values,
                                                   system_matrix,
                                                   solution,
                                                   system_rhs);

                solve_time_step();

                // accetta e adatta dt basandosi su error_est
                time = t_new;
                ++timestep_number;
                old_solution = solution;
                std::cout << "Heuristic step " << timestep_number << " at t=" << time
                          << "  dt=" << dt << "  err_est=" << error_est << std::endl;
                output_results();

                if ((timestep_number == 1) &&
                        (pre_refinement_step < n_adaptive_pre_refinement_steps) && use_space_adaptivity)
                      {
                        refine_mesh(initial_global_refinement,
                                    initial_global_refinement +
                                      n_adaptive_pre_refinement_steps);
                        ++pre_refinement_step;

                        std::cout << std::endl;

                        goto start_time_iteration;
                      }
                    else if ((timestep_number > 0) && (timestep_number % 5 == 0) && use_space_adaptivity)
                      {
                        refine_mesh(initial_global_refinement,
                                    initial_global_refinement +
                                      n_adaptive_pre_refinement_steps);
                        tmp.reinit(solution.size());
                        forcing_terms.reinit(solution.size());
                      }

                const double err_ratio = (error_est > 0.0) ? (tol_scaled / error_est) : 1e6;
                double dt_new = dt * time_step_safety * std::pow(err_ratio, 1.0 / (p + 1.0));
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

      bool mesh = ask_bool("Vuoi generare una mesh o importarla da file?");
      int cells_per_direction = 0; // valore fittizio, non usato
      if (mesh)
          {
            std::cout << "Inserisci il numero di celle per direzione (intero positivo): ";
            std::cin >> cells_per_direction;
            if (cells_per_direction <= 0)
              {
                std::cerr << "Numero di celle non valido. Deve essere un intero positivo." << std::endl;
                return 1;
              }
            std::cout << "Generazione di una mesh con " << cells_per_direction << " celle per direzione." << std::endl;
          }
      else
          {
            cells_per_direction = 0; // valore fittizio, non usato
          }
      bool space = ask_bool("Vuoi abilitare l'adattività spaziale?");
      bool time = ask_bool("Vuoi abilitare l'adattività temporale?");
      if (time)
          {
            bool type = ask_bool("Se sì, vuoi usare lo step-doubling? (altrimenti verrà usata l'euristica economica)");
 
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
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}