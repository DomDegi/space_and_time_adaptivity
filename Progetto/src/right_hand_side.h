#ifndef RIGHT_HAND_SIDE_H
#define RIGHT_HAND_SIDE_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/numbers.h>
#include <cmath>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const unsigned int N_in,
                const double sigma_in,
                const double a_in,
                const Point<dim> &x0_in)
    : Function<dim>()
    , N(N_in)
    , sigma(sigma_in)
    , a(a_in)
    , x0(x0_in)
  {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  const unsigned int N;
  const double sigma;
  const double a;
  const Point<dim> x0;
};

// Implementation inline
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const
{
  (void)component;
  AssertIndexRange(component, 1);
  // This logic is specific to 2D in the original code, but kept generic structure
  // Assert(dim == 2, ExcNotImplemented()); 

  const double time = this->get_time();
  const double g = std::exp(-a * std::cos(2.0 * N * numbers::PI * time)) / std::exp(a);

  double r2 = 0.0;
  for (unsigned int d = 0; d < dim; ++d)
  {
    const double diff = p[d] - x0[d];
    r2 += diff * diff;
  }

  const double h = std::exp(-r2 / (sigma * sigma));
  return g * h;
}

#endif // RIGHT_HAND_SIDE_H
