#ifndef FC_HPP
#define FC_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <cassert>

//#include "cube2vector.hpp"

#define DEBUG false
#define DEBUG_PREFIX "[DEBUG DENSE LAYER ]\t"

class FC
{
 public:
  FC(size_t numInputs,size_t numOutputs) :
      numInputs(numInputs),
      numOutputs(numOutputs)
  {
    // Initialize the weights.
    weights = arma::zeros(numOutputs, numInputs);
    weights.imbue( [&]() { return _getTruncNormalVal(0.0, 1.0); } );

    // Initialize the biases
    //biases = arma::vec(numOutputs ,arma::fill::randu);
    biases = arma::vec(numOutputs);
    //std::cout<<"biases: "<<biases<<std::endl;
    //mat A(5, 5, fill::randu);

    // Reset accumulated gradients.
    _resetAccumulatedGradients();
  }


  void Forward(arma::vec& input, arma::vec& output)
  {
    //arma::vec flatInput = input;
    output = (weights * input) + biases;

    this->input = input;
    this->output = output;
  }

  void Backward(arma::vec& upstreamGradient)
  {
    gradInputVec = arma::zeros(numInputs);
    for (size_t i=0; i<(numInputs); i++)
      gradInputVec[i] = arma::dot(weights.col(i), upstreamGradient);
    arma::vec tmp(numInputs);
    tmp = gradInputVec;
    //gradInput = this->tmp;
    //tmp.slice(0).col(0) = gradInputVec;
    //gradInput = arma::reshape(tmp, inputHeight, inputWidth, inputDepth);

    //std::cout<<"ecc: "<<accumulatedGradInput<<std::endl;
    //std::cout<<"gradInput: "<<gradInput<<std::endl;

    accumulatedGradInput += gradInputVec;

    gradWeights = arma::zeros(arma::size(weights));
    for (size_t i=0; i<gradWeights.n_rows; i++)
      gradWeights.row(i) = input.t() * upstreamGradient[i];

    accumulatedGradWeights += gradWeights;

    gradBiases = upstreamGradient;
    accumulatedGradBiases += gradBiases;
  }

  void UpdateWeightsAndBiases(size_t batchSize, double learningRate)
  {
    weights = weights - learningRate * (accumulatedGradWeights/batchSize);
    biases = biases - learningRate * (accumulatedGradBiases/batchSize);
    _resetAccumulatedGradients();
  }

  arma::mat getGradientWrtWeights() { return gradWeights; }

  arma::vec getGradientWrtInput() { return gradInputVec; }

  arma::vec getGradientWrtBiases() { return gradBiases; }

  arma::mat getWeights() { return weights; }

  arma::vec getBiases() { return biases; }

  void setWeights(arma::mat weights) { this->weights = weights; }

  void setBiases(arma::vec biases) { this->biases = biases; }

 private:
//   size_t inputHeight;
//   size_t inputWidth;
//   size_t inputDepth;
  arma::vec input;
  size_t numInputs;
  size_t numOutputs;
  arma::vec output;

  arma::mat weights;
  arma::vec biases;

  arma::vec gradInputVec;
  arma::mat gradWeights;
  arma::vec gradBiases;

  arma::mat accumulatedGradInput;
  arma::mat accumulatedGradWeights;
  arma::vec accumulatedGradBiases;

  double _getTruncNormalVal(double mean, double variance)
  {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
      candidate.randn(1, 1);
    return candidate[0];
  }

  void _resetAccumulatedGradients()
  {
    accumulatedGradInput = arma::zeros(numInputs);
    accumulatedGradWeights = arma::zeros(
        numOutputs,
        numInputs
        );
    accumulatedGradBiases = arma::zeros(numOutputs);
  }
};

#undef DEBUG
#undef DEBUG_PREFIX

#endif
