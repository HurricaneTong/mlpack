/**
 * @file quadric_function.hpp
 * @author Shangtong Zhang
 *
 * Definition and implementation of the quadric function.
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUADRIC_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_QUADRIC_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<double Power>
class QuadricFunction {
  
 public:
  
  QuadricFunction(double factor) :
  factor(factor) {
    
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
    return std::pow(1 + tmp * tmp, Power);
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

typedef QuadricFunction<0.5> MultiQuadricFunction;
typedef QuadricFunction<-1> InverseQuadricFunction;
typedef QuadricFunction<-0.5> InverseMultiQuadricFunction;
  
}; // namespace ann
}; // namespace mlpack

#endif
