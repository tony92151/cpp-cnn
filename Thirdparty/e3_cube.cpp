#include "../layers/convolution_layer.hpp"
#include "../layers/max_pooling_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/tanh_layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/cross_entropy_loss_layer.hpp"
#include "../utils/mnist.hpp"

#include "../layers/cube2vector.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <time.h> 

int main(int argc, char ** argv){
    // arma::cube x(1,2,3);
    // arma::cube y = arma::randu<arma::cube>(4,5,6);

    // arma::mat A = y.slice(1);  // extract a slice from the cube
    //                     // (each slice is a matrix)

    // arma::mat B = arma::randu<arma::mat>(4,5);
    // y.slice(2) = B;     // set a slice in the cube

    // arma::cube q = y + y;     // cube addition
    // arma::cube r = y % y;     // element-wise cube multiplication

    // std::cout<< x <<std::endl;
    // std::cout<< q <<std::endl;

    // std::cout<< r <<std::endl;

    // arma::cube::fixed<4,5,6> f;
    // f.ones();

    arma::cube x = arma::zeros(1,1,4);

    arma::mat b = x.slice(2);

    std::cout<< b(0,0) <<std::endl;

    x(1) = (double)0.905;

    std::cout<< x <<std::endl;

    arma::vec c = arma::zeros(x.size());

    cube2vector(x,c);

    std::cout<< arma::vectorise(x) <<std::endl;


}
