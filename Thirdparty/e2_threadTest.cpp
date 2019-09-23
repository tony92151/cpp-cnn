// #include <thread>
// #include <iostream>
// #include <string>
// #include <vector>
// #include<pthread.h>

// void func(int i, string s)
// {
//     cout << i << ", " << this_thread::get_id() << endl;
// }

// int main()
// {

//   arma::cube cub = arma::zeros(24, 24, 6);

//   std::cout<<"cube before : "<<cub<<std::endl;






//     std::vector<std::thread> threads;
//     for(int i = 0; i < 10; i++){
//         threads.push_back(std::thread(func, i, "test"));
//     }   
//     for(int i = 0; i < threads.size(); i++){
//         cout << threads[i].get_id() << endl;
//         threads[i].join();
//     }   
//     return 0;
// }

#include <armadillo>
//#include <boost/test/unit_test.hpp>
#include <iostream>
#include <thread>


void newThreadCallback(arma::cube& p)
{
    std::cout<<"Inside Thread :  "" : p = "<<p<<std::endl;
    std::chrono::milliseconds dura( 1000 );
    std::this_thread::sleep_for( dura );
    std::cout<<"Inside out Thread :  = "<< p <<std::endl;
    p = arma::zeros(1, 3, 1);
}

class test{
  public:
  arma::cube cub = arma::zeros(3, 3, 1);
  void tran(arma::cube& cu){
    int i = 10;
    //arma::cube cub = arma::zeros(3, 3, 1);

    std::cout<<"Inside Main Thread :  = "<< cu <<std::endl;
    std::thread t(newThreadCallback,std::ref(cu));
    t.join();
    std::cout<<"Inside Main Thread :  = "<< cu <<std::endl;
  }
};



void startNewThread(arma::cube& input)
{
    int i = 10;
    //arma::cube cub = arma::zeros(3, 3, 1);

    std::cout<<"Inside Main Thread :  = "<< input <<std::endl;
    std::thread t(newThreadCallback,std::ref(input));
    t.join();
    std::cout<<"Inside Main Thread :  = "<< input <<std::endl;
}
int main()
{
  arma::cube cub = arma::zeros(3, 3, 1);
    //startNewThread(cub);
    //std::chrono::milliseconds dura( 2000 );
    //std::this_thread::sleep_for( dura );

    test T;
    T.tran(cub);
    return 0;
}