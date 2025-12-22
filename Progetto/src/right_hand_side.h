/**
 * @file right_hand_side.h
 * @brief Defines the source term function f(x, t) for the heat equation.
 */

#ifndef RIGHT_HAND_SIDE_H
#define RIGHT_HAND_SIDE_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/numbers.h>
#include <cmath>

using namespace dealii;

/**
 * @class RightHandSide
 * @brief A time-dependent spatial function used as the forcing term.
 *
 * Implements a Gaussian source that pulsates in time.
 * Mathematical form: f(x, t) = g(t) * h(x)
 * * where:
 * - h(x) = exp(-|x-x0|^2 / sigma^2)  (Spatial Gaussian)
 * - g(t) = exp(-a * cos(2*pi*N*t)) / exp(a) (Temporal oscillation)
 * * @tparam dim The spatial dimension (e.g., 2 or 3).
 */
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  /**
   * @brief Constructor.
   * @param N_in Frequency parameter for the time oscillation.
   * @param sigma_in Width (standard deviation related) of the spatial Gaussian.
   * @param a_in Parameter controlling the magnitude/sharpness of time oscillation.
   * @param x0_in The spatial center of the Gaussian source.
   */
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

  /**
   * @brief Evaluates the function at a given point and current time.
   * * @param p The point in space.
   * @param component The vector component (always 0 for scalar heat equation).
   * @return The computed value of the source term.
   */
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  const unsigned int N;   ///< Frequency of oscillation
  const double sigma;     ///< Spatial width
  const double a;         ///< Amplitude/decay parameter
  const Point<dim> x0;    ///< Center of the source
};

// --- Implementation ---

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const
{
  (void)component;
  AssertIndexRange(component, 1);
  // Note: While primarily designed for 2D, this math works generically for dim=1,2,3.

  const double time = this->get_time();

  // Temporal part: g(t)
  const double g = std::exp(-a * std::cos(2.0 * N * numbers::PI * time)) / std::exp(a);

  // Spatial part: h(x) = Gaussian
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