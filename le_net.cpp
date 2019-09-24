#include "layers/convolution_layer.hpp"
#include "layers/max_pooling_layer.hpp"
#include "layers/relu_layer.hpp"
#include "layers/tanh_layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/fc.hpp"
#include "layers/cross_entropy_loss_layer.hpp"
#include "utils/mnist.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <fstream>

#include <time.h>

#define DEBUG true
#define DEBUG_PREFIX "[DEBUG LE NET ]\t"

int main(int argc, char ** argv)
{

    std::string dataPath = "/home/ros/Documents/cpp-cnn/data";
    std::string savePath;
    size_t EP = 5;
    double lr = 0.001;
    std::cout<<"==============================="<<std::endl;


    for (int i = 1; i < argc; i++) {

        //std::cout<<"argc : "<<i<<": "<<argv[i]<<std::endl;

        if (strcmp(argv[i],"-d")==0) {
            dataPath = argv[i+1];
            std::cout<<"Data path : "<<dataPath<<std::endl;
        } else if (strcmp(argv[i],"-c")==0) {
            savePath = argv[i + 1];
            std::cout<<"Save Path : "<<savePath<<std::endl;
        } else if (strcmp(argv[i],"-e")==0) {
            EP = atoi(argv[i + 1]);
            std::cout<<"EPOCHS : "<<EP<<std::endl;
        } else if (strcmp(argv[i],"-l")==0) {
            lr = atof(argv[i + 1]);
            std::cout<<"LEARNING RATE : "<<lr<<std::endl;
        }else{
            //std::cout<<"Error in arg."<<std::endl;
            //return 0;
        }

    }
    std::cout<<"==============================="<<std::endl;

//std::cout<<"Save Path : "<<savePath<<std::endl;
  // Read the Kaggle data
 // MNISTData md("../data");
 //MNISTData md("/home/ros/Documents/cpp-cnn/data");
 MNISTData md(dataPath);

  std::vector<arma::cube> trainData = md.getTrainData();
  std::vector<arma::vec> trainLabels = md.getTrainLabels();

  std::vector<arma::cube> validationData = md.getValidationData();
  std::vector<arma::vec> validationLabels = md.getValidationLabels();

  assert(trainData.size() == trainLabels.size());
  assert(validationData.size() == validationLabels.size());

  std::vector<arma::cube> testData = md.getTestData();

#if DEBUG
  std::cout << DEBUG_PREFIX
      << "Training data size: " << trainData.size() << std::endl;
  std::cout << DEBUG_PREFIX
      << "Validation data size: " << validationData.size() << std::endl;
  std::cout << DEBUG_PREFIX
      << "Test data size: " << testData.size() << std::endl;
  std::cout << DEBUG_PREFIX << std::endl;
#endif

  const size_t TRAIN_DATA_SIZE = trainData.size();
  const size_t VALIDATION_DATA_SIZE = validationData.size();
  const size_t TEST_DATA_SIZE = testData.size();
   double LEARNING_RATE = lr;
   size_t EPOCHS = EP;
  const size_t BATCH_SIZE = 100;
  const size_t NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

  // Define the network layers
  ConvolutionLayer c1(
      28,
      28,
      1,
      3,
      3,
      1,
      1,
      12);//12
  // Output is 26 x 26 x 12

//   ReLULayer r1(
//       24,
//       24,
//       6);
  Tanh r1(
      26,
      26,
      12);
  // Output is 26 x 26 x 12
  ConvolutionLayer c11(
      26,
      26,
      12,
      3,
      3,
      1,
      1,
      24);
    Tanh r11(
      24,
      24,
      24);

  MaxPoolingLayer mp1(
      24,
      24,
      24,
      2,
      2,
      2,
      2);
  // Output is 12 x 12 x 6

  ConvolutionLayer c2(
      12,
      12,
      24,
      5,
      5,
      1,
      1,
      24);
  // Output is 8 x 8 x 16

  ReLULayer r2(
      8,
      8,
      24);
  // Output is 8 x 8 x 16

  MaxPoolingLayer mp2(
      8,
      8,
      24,
      2,
      2,
      2,
      2);
  // Output is 4 x 4 x 16

  DenseLayer d(
      4,
      4,
      24,
      128);

  FC fc1(
      128,
      32);
  FC fc2(
      32,
      10);
  // Output is a vector of size 10

  SoftmaxLayer s(10);
  // Output is a vector of size 10

  CrossEntropyLossLayer l(10);

  // Initialize armadillo structures to store intermediate outputs (Ie. outputs
  // of hidden layers)
  arma::cube c1Out = arma::zeros(26, 26, 12);
  arma::cube r1Out = arma::zeros(26, 26, 12);

  arma::cube c11Out = arma::zeros(26, 26, 24);
  arma::cube r11Out = arma::zeros(24, 24, 24);

  arma::cube mp1Out = arma::zeros(12, 12, 24);
  arma::cube c2Out = arma::zeros(12, 12, 24);
  arma::cube r2Out = arma::zeros(8, 8, 24);
  arma::cube mp2Out = arma::zeros(4, 4, 24);
  arma::vec dOut = arma::zeros(128);
  arma::vec fcOut = arma::zeros(32);
  arma::vec fc2Out = arma::zeros(10);
  arma::vec sOut = arma::zeros(10);

  // Initialize loss and cumulative loss. Cumulative loss totals loss over all
  // training examples in a minibatch.
  double loss = 0.0;
  double cumLoss = 0.0;

  time_t start, end;
  start = time(NULL);

  std::fstream fout2(savePath,std::ios::out);

  for (size_t epoch = 0; epoch < EPOCHS; epoch++)
  {
#if DEBUG
    //std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Epoch # (" << epoch+1 <<" / "<<EPOCHS<<")"<< std::endl;
#endif
    for (size_t batchIdx = 0; batchIdx < NUM_BATCHES; batchIdx++)
    {
      // Generate a random batch.
      arma::vec batch(BATCH_SIZE, arma::fill::randu);
      batch *= (TRAIN_DATA_SIZE - 1);
      LEARNING_RATE*=0.95;

      for (size_t i = 0; i < BATCH_SIZE; i++)
      {
          if((BATCH_SIZE*batchIdx+i)%500==0){
              std::cout << "STEP: ("<< BATCH_SIZE*batchIdx+i <<"/"<< trainData.size() <<")" << std::endl;
          }
        // Forward pass
        //std::cout<<"brfore: "<<c1Out<<std::endl;
        c1.Forward(trainData[batch[i]], c1Out);
        //std::cout<<"after: "<<c1Out<<std::endl;
        r1.Forward(c1Out, r1Out);

        c11.Forward(r1Out,c11Out);
        r11.Forward(c11Out,r11Out);

        mp1.Forward(r11Out, mp1Out);
        //std::cout<<"c2"<<std::endl;
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        fc1.Forward(dOut,fcOut);
        fc2.Forward(fcOut,fc2Out);
        s.Forward(fc2Out, sOut);

        // Compute the loss
        loss = l.Forward(sOut, trainLabels[batch[i]]);
        cumLoss += loss;

        // Backward pass
        l.Backward();
        arma::vec gradWrtPredictedDistribution =
            l.getGradientWrtPredictedDistribution();
        s.Backward(gradWrtPredictedDistribution);
        arma::vec gradWrtSIn = s.getGradientWrtInput();

        fc2.Backward(gradWrtSIn);
        arma::vec gradWrtFC2In = fc2.getGradientWrtInput();

        fc1.Backward(gradWrtFC2In);
        arma::vec gradWrtFCIn = fc1.getGradientWrtInput();

        d.Backward(gradWrtFCIn);
        arma::cube gradWrtDIn = d.getGradientWrtInput();
        mp2.Backward(gradWrtDIn);
        arma::cube gradWrtMP2In = mp2.getGradientWrtInput();
        r2.Backward(gradWrtMP2In);
        arma::cube gradWrtR2In = r2.getGradientWrtInput();
        //std::cout<<"c2B"<<std::endl;
        c2.Backward(gradWrtR2In);
        arma::cube gradWrtC2In = c2.getGradientWrtInput();
        mp1.Backward(gradWrtC2In);
        arma::cube gradWrtMP1In = mp1.getGradientWrtInput();

        r11.Backward(gradWrtMP1In);
        arma::cube gradWrtR11In = r11.getGradientWrtInput();
        //std::cout<<"c1B"<<std::endl;
        c11.Backward(gradWrtR11In);
        arma::cube gradWrtC11In = c11.getGradientWrtInput();


        r1.Backward(gradWrtC11In);
        arma::cube gradWrtR1In = r1.getGradientWrtInput();
        //std::cout<<"c1B"<<std::endl;
        c1.Backward(gradWrtR1In);
        arma::cube gradWrtC1In = c1.getGradientWrtInput();
      }

      // Update params

      d.UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
      fc1.UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
      fc2.UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
      c1.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      c11.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
      c2.UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
    }

#if DEBUG
    // Output loss on training dataset after each epoch
    std::cout << DEBUG_PREFIX << std::endl;
    std::cout << DEBUG_PREFIX << "Training loss: "
        << cumLoss / (BATCH_SIZE * NUM_BATCHES) << std::endl;
#endif

    // Compute the training accuracy after epoch
    double correct = 0.0;
    for (size_t i = 0; i < TRAIN_DATA_SIZE; i++)
    {
      // Forward pass
      //std::cout<<"c1"<<std::endl;
      c1.Forward(trainData[i], c1Out);
        r1.Forward(c1Out, r1Out);

        c11.Forward(r1Out,c11Out);
        r11.Forward(c11Out,r11Out);

        mp1.Forward(r11Out, mp1Out);
        //std::cout<<"c2"<<std::endl;
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        fc1.Forward(dOut,fcOut);
        fc2.Forward(fcOut,fc2Out);
        s.Forward(fc2Out, sOut);

      if (trainLabels[i].index_max() == sOut.index_max())
        correct += 1.0;
    }

#if DEBUG
    // Output accuracy on training dataset after each epoch
    std::cout << DEBUG_PREFIX
        << "Training accuracy: " << correct/TRAIN_DATA_SIZE << std::endl;
#endif

    // Compute validation accuracy after epoch
    cumLoss = 0.0;
    correct = 0.0;
    for (size_t i = 0; i < VALIDATION_DATA_SIZE; i++)
    {
      // Forward pass
      c1.Forward(validationData[i], c1Out);
        r1.Forward(c1Out, r1Out);

        c11.Forward(r1Out,c11Out);
        r11.Forward(c11Out,r11Out);

        mp1.Forward(r11Out, mp1Out);
        //std::cout<<"c2"<<std::endl;
        c2.Forward(mp1Out, c2Out);
        r2.Forward(c2Out, r2Out);
        mp2.Forward(r2Out, mp2Out);
        d.Forward(mp2Out, dOut);
        dOut /= 100;
        fc1.Forward(dOut,fcOut);
        fc2.Forward(fcOut,fc2Out);
        s.Forward(fc2Out, sOut);

      cumLoss += l.Forward(sOut, validationLabels[i]);

      if (validationLabels[i].index_max() == sOut.index_max())
        correct += 1.0;
    }

#if DEBUG
    // Output validation loss after each epoch
    std::cout << DEBUG_PREFIX
        << "Validation loss: " << cumLoss / (BATCH_SIZE * NUM_BATCHES)
        << std::endl;

    // Output validation accuracy after each epoch
    std::cout << DEBUG_PREFIX
        << "Val accuracy: " << correct / VALIDATION_DATA_SIZE << std::endl;
    std::cout << DEBUG_PREFIX << std::endl;
#endif

    // Reset cumulative loss and correct count
    cumLoss = 0.0;
    correct = 0.0;

//std::cout << DEBUG_PREFIX << "Epoch # (" << epoch+1 <<" / "<<EPOCHS<<")"<< std::endl;
    fout2 << DEBUG_PREFIX << "Epoch # ("+std::to_string(epoch+1)+" / "<<std::to_string(EPOCHS)<<")"<< std::endl;
    fout2 << DEBUG_PREFIX << std::endl;
    fout2 << DEBUG_PREFIX << "Training loss: "+std::to_string(cumLoss / (BATCH_SIZE * NUM_BATCHES)) << std::endl;
    fout2 << DEBUG_PREFIX
        << "Training accuracy: "+std::to_string(correct/TRAIN_DATA_SIZE) << std::endl;

    fout2 << DEBUG_PREFIX
        << "Validation loss: "+std::to_string(cumLoss / (BATCH_SIZE * NUM_BATCHES))
        << std::endl;


    fout2 << DEBUG_PREFIX
        << "Val accuracy: "+std::to_string(correct / VALIDATION_DATA_SIZE) << std::endl;
    fout2 << DEBUG_PREFIX << std::endl;

    //fout << "ImageId,Label" << std::endl;
    //fout << "ImageId,Label" << std::endl;

    // Write results on test data to results csv
    // std::fstream fout(savePath+"/results_epoch_" + std::to_string(epoch) + ".csv",
    //                   std::ios::out);
    // fout << "ImageId,Label" << std::endl;
    // for (size_t i=0; i<TEST_DATA_SIZE; i++)
    // {
    //   // Forward pass
    //   c1.Forward(testData[i], c1Out);
    //   r1.Forward(c1Out, r1Out);
    //   mp1.Forward(r1Out, mp1Out);
    //   c2.Forward(mp1Out, c2Out);
    //   r2.Forward(c2Out, r2Out);
    //   mp2.Forward(r2Out, mp2Out);
    //   d.Forward(mp2Out, dOut);
    //   dOut /= 100;
    //   fc1.Forward(dOut,fcOut);
    //   s.Forward(fcOut, sOut);
    //   //s.Forward(dOut, sOut);

    //   fout << std::to_string(i+1) << ","
    //       << std::to_string(sOut.index_max()) << std::endl;
    // }
    // fout.close();
  }

  


  end = time(NULL);
  double diff = difftime(end, start);
  std::cout<<"Time = "<<diff<< "s" <<std::endl;
  fout2<<"Time = "<<diff<< "s" <<std::endl;

  fout2.close();
}


#undef DEBUG
#undef DEBUG_PREFIX
