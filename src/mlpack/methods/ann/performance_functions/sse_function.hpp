/**
 * @file sse_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the sum squared error performance function.
 */
#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SSE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The sum squared error performance function measures the network's performance
 * according to the sum of squared errors.
 *
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template<typename VecType = arma::colvec>
class SumSquaredErrorFunction
{
  public:
  /**
   * Computes the sum squared error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return sum of squared errors.
   */
  static double Error(const VecType& input, const VecType& target)
  {
    return arma::sum(arma::square(target - input));
  }

}; // class SumSquaredErrorFunction

}; // namespace ann
}; // namespace mlpack

#endif
