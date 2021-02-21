/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFANNLAYER_H
#define MFANNLAYER_H

//#include <MfMatrix.h>
#include <random>
#include <string>
#include <Malloc.h>
#include <Math.h>
#include <iostream>
#include <stdio.h>

#include "MfDoubleMatrix.h"
#include "Activator.h"
using namespace std;

class MfAnnLayer
{
    public:
        //The default constructor
        MfAnnLayer();

        //Constructor for input/output size
        MfAnnLayer(int paraInputSize, int paraOutputSize, char paraActivation,
            double paraRate, double paraMobp);

        //Destructor
        virtual ~MfAnnLayer();

        //Convert to string for display
        string toString();

        //Show weights
        void showWeight();

        //Set the activation function
        void setActivationFunction(char paraActivation)
        {
            activator->setActivationFunction(paraActivation);
        }

        //Getter.
        Activator* getActivator()
        {
            return activator;
        }

        //Reset weight and other variables.
        void reset();

        //Getter
        int getInputSize()
        {
            return inputSize;
        }

        //Getter
        int getOutputSize()
        {
            return outputSize;
        }

        //Forward calculation
        MfDoubleMatrix* forward(MfDoubleMatrix* paraData);

        //Back propagation calculation
        MfDoubleMatrix* backPropagation(MfDoubleMatrix* paraLabel);

        //Code unit test
        void unitTest();

    protected:

        //The size of the input
        int inputSize;

        //The size of the output
        int outputSize;

        //Learning rate.
        double learningRate;

        //Mobp
        double mobp;

        //The activator
        Activator* activator;

        //The input data, only one row
        MfDoubleMatrix* inputData;

        //The output data, only one row
        MfDoubleMatrix* outputData;

        //The weights for edges
        MfDoubleMatrix* weightMatrix;

        //The weight change for edges
        MfDoubleMatrix* weightDeltaMatrix;

        //The offset
        MfDoubleMatrix* offsetMatrix;

        //The offset delta
        MfDoubleMatrix* offsetDeltaMatrix;

        //The layer node error
        MfDoubleMatrix* errorMatrix;

    private:
};

#endif // ANNLAYER_H
