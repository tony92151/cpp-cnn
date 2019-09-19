#include "../layers/convolution_layer.hpp"
#include "../layers/max_pooling_layer.hpp"
#include "../layers/relu_layer.hpp"
#include "../layers/tanh_layer.hpp"
#include "../layers/dense_layer.hpp"
#include "../layers/softmax_layer.hpp"
#include "../layers/cross_entropy_loss_layer.hpp"
#include "../utils/mnist.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <time.h> 

using namespace std;

int main(int argc, char ** argv){
    srand( time(NULL) );

    // if (argc-1==0) {
    //     cout<<"Error path to pose.txt file."<<endl;
    //     return 0;
    // }else {
    //     cout<<"Reading file :"<< argv[1] <<endl;
    // }
    // ifstream fin(argv[1]);
    // if(!fin){
    //     cout<<"Reading Error!"<<endl;
    // }else{
    //     cout<<"Reading Success!"<<endl;
    // }
    

    arma::cube A = arma::zeros(10,1,1);
    arma::cube out = arma::zeros(24, 1, 1);

    Tanh T(10,1,1);
    for(int i=0 ;i<10 ;i++){
        //A[i] = (double) (rand() / (RAND_MAX + 1.0))*2 - 1;
        A[i] = (double)i/10;
    }
    
    cout<<"A defore: "<< A <<endl;

    T.Forward(A,out); 

    //A.transform( [](double val) { return val > 0? 1 : 0; } );
    cout<<"A after: "<< out <<endl;
}


