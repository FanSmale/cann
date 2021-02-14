/*
 * The C++ Artificial Neural network project.
 * This class constructs a full ANN using AnnLayers.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFFULLANN_H
#define MFFULLANN_H

#include <iostream>
#include "MfIntArray.h"
#include "MfDoubleMatrix.h"
#include "MfDataReader.h"
#include "MfAnnLayer.h"

class MfFullAnn
{
    public:
        //The default constructor
        MfFullAnn();

        //The Ann with given sizes and activation function
        MfFullAnn(MfIntArray* paraSizes, char paraActivation, double paraRate, double paraMobp);

        //Destructor
        virtual ~MfFullAnn();

        //Convert to string for display
        string toString();

        //Set the activation function for the given layer
        void setActivationFunction(int paraLayer, char paraFunction);

        //Setter
        void setRate(double paraRate);

        //Setter
        void setMobp(double paraMobp);

        //Reset weights and other variables.
        void reset();

        //Forward layer by layer
        MfDoubleMatrix* forward(MfDoubleMatrix* paraInput);

        //Back propagation
        void backPropagation(MfDoubleMatrix* paraTarget);

        //Train the network with only one instance
        void train(MfDoubleMatrix* paraX, int paraY);

        //Train with a dataset
        void train(MfDoubleMatrix* paraX, MfIntArray* paraY);

        //Test with an instance
        bool test(MfDoubleMatrix* paraX, int paraY);

        //Test with a dataset
        double test(MfDoubleMatrix* paraX, MfIntArray* paraY);

        //Get the number of correctly classified instances in the current round
        int getNumCorrect();

        //Show weight of the network, not including the offset
        void showWeight();

        //Code unit test
        void unitTest();

        //Training/testing test
        void trainingTestingTest();

        //Cross validation test
        void crossValidationTest();

    protected:

        //Number of layers
        int numLayers;

        //Learning rate
        double rate;

        //Mobp
        double mobp;

        //Activation
        char activation;

        //The sizes of layers
        MfIntArray* layerSizes;

        //All layers
        MfAnnLayer** layers;

        //The output for current instance
        MfDoubleMatrix* currentOutput;

        //The current instance
        MfDoubleMatrix* currentInstance;

        //The decision of the current instance
        MfDoubleMatrix* currentDecision;

        //Number of correctly classified instances in the current round
        //It is used for statistics especially on cross-validation
        int numCorrect;

    private:
};

#endif // MFFULLANN_H
