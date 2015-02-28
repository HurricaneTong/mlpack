/**
 * @file cee_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the cross-entropy error performance
 * function.
 */
#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/neuron_layer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cross-entropy error performance function measures the network's
 * performance according to the cross entropy errors. The log in the cross-
 * entropy take sinto account the closeness of a prediction and is a more
 * granular way to calculate the error.
 *
 * @tparam Layer The layer that is connected with the output layer.
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template<
    class Layer = NeuronLayer< >,
    typename VecType = arma::colvec
>
class CrossEntropyErrorFunction
{
  public:
  /**
   * Computes the cross-entropy error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return cross-entropy error.
   */
  static double Error(const VecType& input, const VecType& target)
  {
    if (LayerTraits<Layer>::IsBinary)
      return -arma::dot(arma::trunc_log(arma::abs(target - input)), target);

    return -arma::dot(arma::trunc_log(input), target);
  }

}; // class CrossEntropyErrorFunction

}; // namespace ann
}; // namespace mlpack

#endif
