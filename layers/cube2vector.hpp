#ifndef CUBE2VECTOR_HPP
#define CUBE2VECTOR_HPP

#include <iostream>
#include <armadillo>
#include <math.h>

void cube2vector(arma::cube& cubein, arma::vec& output){
    arma::vec vecout = arma::zeros(cubein.size());
    for (int i=0;i<cubein.size();i++){
        arma::mat b = cubein.slice(i);
        vecout(i) = b(0,0);
    }
    output = vecout;
}

void vector2cube(arma::vec& vecin, arma::cube& output){
    arma::cube cubeout = arma::zeros(1,1,vecin.size());
    for (int i=0;i<vecin.size();i++){
        cubeout(i) = vecin(i);
        //vecout(i) = b(0,0);
    }
    output = cubeout;
}

#endif
