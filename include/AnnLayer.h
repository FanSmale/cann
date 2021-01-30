/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef ANNLAYER_H
#define ANNLAYER_H

//#include <MfMatrix.h>
#include <string>
#include "MfMath.h"
#include "Activator.h"

using namespace std;

class AnnLayer
{
    public:
        //The default constructor
        AnnLayer();

        //Constructor for input/output size
        AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation,
                 double paraRate, double paraMobp);

        //Destructor
        virtual ~AnnLayer();

        //Convert to string for display
        string toString();

        //Set the activation function
        void setActivationFunction(char paraActivation);

        //Activate
        //double activate(double paraValue);

        //Forward calculation
        DoubleMatrix forward(DoubleMatrix paraData);

        //Back propagation calculation
        DoubleMatrix backPropagation(DoubleMatrix paraLabel);

        //Code unit test
        void unitTest();

        //Show weights
        void showWeight();

        //The size of the output
        int outputSize;

    protected:

        //The size of the input
        int inputSize;

        //The activation function
        //char activation;

        //The activator
        Activator activator;

        //Learning rate
        double rate;

        //Mobp
        double mobp;

        //Weighted error sum
        double errorSum;

        //The input data, only one row
        DoubleMatrix inputData;

        //The weights for edges
        DoubleMatrix weightMatrix;

        //The weight change for edges
        DoubleMatrix weightDeltaMatrix;

        //The offset
        DoubleMatrix offsetMatrix;

        //The offset delta
        DoubleMatrix offsetDeltaMatrix;

        //The layer node error
        DoubleMatrix errorMatrix;

    private:
};

#endif // ANNLAYER_H
