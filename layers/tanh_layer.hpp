#ifndef TANH_LAYER_HPP
#define TANH_LAYER_HPP

#include <iostream>
#include <armadillo>
#include <math.h>

class Tanh{
    public:

    Tanh(size_t inputHeight,
        size_t inputWidth,
        size_t inputDepth):
        inputHeight(inputHeight),
        inputWidth(inputWidth),
        inputDepth(inputDepth)
      {

      };

    void Forward(arma::cube& input, arma::cube& output){

        output = arma::tanh(input);
 
        this->input = input;
        this->output = output;
    }

    void Backward(arma::cube upstreamGradient){
        //gradientWrtInput = input;
        //gradientWrtInput.transform( [](double val) { return math.pow(arma::sech(val),2); } );
        this->gradientWrtInput = pow(upstreamGradient,2);
    }

    arma::cube getGradientWrtInput() { return gradientWrtInput; }

    private:
    size_t inputHeight;
    size_t inputWidth;
    size_t inputDepth;

    arma::cube input;
    arma::cube output;

    arma::cube gradientWrtInput;
};

#endif