/*
 * The C++ Artificial Neural network project.
 * This class constructs a full ANN using AnnLayers.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef FULLANN_H
#define FULLANN_H

#include "AnnLayer.h"
#include "MfMath.h"

class FullAnn
{
    public:
        //The default constructor
        FullAnn();

        //The Ann with given sizes and activation function
        FullAnn(IntArray paraSizes, char paraActivation, double paraRate, double paraMobp);

        //Destructor
        virtual ~FullAnn();

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
        DoubleMatrix forward(DoubleMatrix paraInput);

        //Back propagation
        void backPropagation(DoubleMatrix paraTarget);

        /**
         * Train the network with only one instance
         * paraX: the instance (1 * m row vector)
         * paraY: the decision of the instance
         * paraNumClasses: the number of classes of this dataset
         */
        void train(DoubleMatrix paraX, int paraY, int paraNumClasses);

        //Train with a dataset
        void train(DoubleMatrix paraX, IntArray paraY, int paraNumClasses);

        //Test with an instance
        bool test(DoubleMatrix paraX, int paraY);

        //Test with a dataset
        double test(DoubleMatrix paraX, IntArray paraY);

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
        IntArray layerSizes;

        //All layers
        AnnLayer** layers;

        //The output for current instance
        DoubleMatrix currentOutput;

        //Number of correctly classified instances in the current round
        //It is used for statistics especially on cross-validation
        int numCorrect;

    private:
};

#endif // FULLANN_H
