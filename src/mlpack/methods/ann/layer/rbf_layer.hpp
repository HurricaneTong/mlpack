/**
 * @file rbf_layer.hpp
 * @author Shangtong Zhang
 *
 */
#ifndef __MLPACK_METHOS_ANN_LAYER_RBF_LAYER_HPP
#define __MLPACK_METHOS_ANN_LAYER_RBF_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard network layer.
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
  class RBFType = GaussianFunction
  typename Metric = EuclideanDistance,
  typename MatType = arma::mat,
  typename VecType = arma::colvec
>
class RBFLayer {
  
 public:
  
  RBFLayer(const size_t inputSize, const size_t outputSize) :
  inputSize(inputSize), outputSize(outputSize),
  centroids(inputSize, outputSize),
  distances(outputSize) {
    
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param inputActivation Input data used for evaluating the specified
   * activity function.
   * @param outputActivation Data to store the resulting output activation.
   */
  void FeedForward(const VecType& inputActivation, VecType& outputActivation) {
    for (size_t i = 0; i < centroids.n_cols; ++i) {
      distances(i) = Metric::Evaluate(inputActivation, centroids.unsafe_col(i));
    }
    outputActivation = RBFType::fn(distances);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param inputActivation Input data used for calculating the function f(x).
   * @param error The backpropagated error.
   * @param delta The calculating delta using the partial derivative of the
   * error with respect to a weight.
   */
  void FeedBackward(const VecType& /* not used */,
                    const VecType& error,
                    VecType& delta)
  {
    VecType derivative;
    ActivationFunction::deriv(distances, derivative);
    double err = arma::accu(error % derivative);
    delta.fill(err);
    
  }

  //! Get the input activations.
  VecType& InputActivation() const { return inputActivations; }
  //  //! Modify the input activations.
  VecType& InputActivation() { return inputActivations; }

  //! Get the detla.
  VecType& Delta() const { return delta; }
 //  //! Modify the delta.
  VecType& Delta() { return delta; }
  
  VecType& Distances() const { return distances; }
  VecType& Distances() { return distances; }

  MatType& Centroids() const { return centroids; }
  MatType& Centroids() { return centroids; }
  
  //! Get input size.
  size_t InputSize() const { return inputSize; }
  //  //! Modify the delta.
  size_t& InputSize() { return inputSize; }

  //! Get output size.
  size_t OutputSize() const { return outputSize; }
  //! Modify the output size.
  size_t& OutputSize() { return outputSize; }

 private:
  //! Locally-stored input activation object.
  VecType inputActivations;

  //! Locally-stored delta object.
  VecType delta;
  VecType distances;

  size_t inputSize;
  size_t outputSize;
  
  MatType centroids;
}; // class NeuronLayer



}; // namespace ann
}; // namespace mlpack

#endif
