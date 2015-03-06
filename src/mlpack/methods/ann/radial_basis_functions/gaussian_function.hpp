/**
 * @file gaussian_function.hpp
 * @author Shangtong Zhang
 *
 * Definition and implementation of the gaussian function.
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_GAUSSIAN_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class GaussianFunction {
  
 public:
  
  GaussianFunction(double factor) : factor(factor) {
    
  }
  
  /**
   * Computes the logistic function.
   *
   * @param x Input data.
   * @return f(x).
   */
  template<typename eT>
  double fn(const eT x) {
    double tmp = factor * x;
    return std::exp(-tmp * tmp);
  }

  /**
   * Computes the logistic function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  void fn(const InputVecType& x, OutputVecType& y) {
    y = x;
    y.transform( [](double x) { return fn(x); } );
  }

  /**
   * Computes the first derivative of the logistic function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double deriv(const double y) {
    return y * (1.0 - y);
  }

  /**
   * Computes the first derivatives of the logistic function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  void deriv(const InputVecType& y, OutputVecType& x) {
    x = y % (1.0 - y);
  }
  
 private:
  double factor;
  
}; // class LogisticFunction

}; // namespace ann
}; // namespace mlpack

#endif
