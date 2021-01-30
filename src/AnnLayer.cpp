/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <Malloc.h>
#include <Math.h>
#include <iostream>
#include <stdio.h>
#include "AnnLayer.h"
#include "MfMath.h"
#include "Activator.h"

using namespace Eigen;

/**
 * The default constructor
 */
AnnLayer::AnnLayer()
{
    inputSize = 1;
    outputSize = 1;
}//Of the default constructor

/**
 * Constructor for input/output size
 */
AnnLayer::AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation,
                   double paraRate, double paraMobp)
{
    printf("AnnLayer constructor\r\n");
    inputSize = paraInputSize;
    outputSize = paraOutputSize;
    rate = paraRate;
    mobp = paraMobp;

    weightMatrix.resize(paraInputSize, paraOutputSize);
    weightDeltaMatrix.resize(paraInputSize, paraOutputSize);
    errorMatrix.resize(1, paraInputSize);

    for(int i = 0; i < paraInputSize; i ++)
    {
        for (int j = 0; j < paraOutputSize; j ++)
        {
            weightMatrix(i, j) = random();
        }//Of for j
    }//Of for i

    //weightMatrix.setOnes();
    printf("Matrix resized \r\n");
    offsetMatrix.resize(1, paraOutputSize);
    offsetDeltaMatrix.resize(1, paraOutputSize);
    for (int i = 0; i < paraOutputSize; i ++)
    {
        offsetMatrix(0, i) = random();
    }//Of for i

    //activation = paraActivation;
    //activator = new Activator(paraActivation);
    activator.setActivationFunction(paraActivation);
}//Of the second constructor

/**
 * Destructor.
 */
AnnLayer::~AnnLayer()
{
}//Of the destructor

/**
 * Convert to string for display.
 * Returns: The string showing myself.
 */
string AnnLayer::toString()
{
    string resultString = "I am an ANN layer with size " + to_string(inputSize)
                          + "*" + to_string(outputSize) + "\r\n";
    resultString += "weight matrix: \r\n";
    //resultString += to_string(weightMatrix);

    //resultString += "offset matrix: \r\n";
    //resultString += offsetMatrix -> toString();
    //resultString += "weights ends. \r\n";

    return resultString;
}//Of toString

/**
 * Set the activation function.
 * paraFunction: the activation function in char.
 */
void AnnLayer::setActivationFunction(char paraFunction)
{
    activator.setActivationFunction(paraFunction);
}//Of setActivationFunction

/**
 * Forward computing.
 * paraData: now only one instance is supported.
 */
DoubleMatrix AnnLayer::forward(DoubleMatrix paraData)
{
    //printf("Forwarding, the data is: \r\n");
    //printf(paraData -> toString().data());

    //printf("The weights are: \r\n");
    //printf(weightMatrix -> toString().data());
    inputData = paraData;

    //printf("paraData: \r\n");
    //cout << paraData << endl;
    //printf("weights: \r\n");
    //cout << weightMatrix << endl;
    DoubleMatrix resultData = paraData * weightMatrix;
    //printf("After weighted sum: \r\n");
    //cout << resultData << endl;
    resultData += offsetMatrix;
    //printf("After adding offset: \r\n");
    //cout << resultData << endl;
    //resultData -> activate(activation);
    //printf("before activate:\r\n");
    //cout << resultData(1, 0) << endl;
    //cout << resultData << endl;

    for(int i = 0; i < resultData.cols(); i ++)
    {
        resultData(0, i) = activator.activate(resultData(0, i));
    }//Of for i

    //printf("After activate: \r\n");
    //cout << resultData << endl;

    return resultData;
}//Of forward

/**
 * Back propagation.
 *   The activation function of the last layer is considered.
 * paraErrors: the error at the output end.
 * Returns: the error at the input end (attention: activation function).
 */
DoubleMatrix AnnLayer::backPropagation(DoubleMatrix paraErrors)
{
    for(int i = 0; i < inputSize; i ++)
    {
        double errorSum = 0;

        //Weights adjusting
        for(int j = 0; j < outputSize; j ++)
        {
            errorSum += paraErrors(0, j) * weightMatrix(i, j);
            weightDeltaMatrix(i, j) = mobp * weightDeltaMatrix(i, j)
                                      + rate * paraErrors(0, j) * inputData(0, i);
            weightMatrix(i, j) += weightDeltaMatrix(i, j);
            //printf("%d, %d weightMatrix(i, j) += weightDeltaMatrix(i, j) %lf; = %lf\r\n",
            //      i, j, weightMatrix(i, j), weightDeltaMatrix(i, j));

            if (i == inputSize - 1)
            {
                // Offset adjusting
                offsetDeltaMatrix(0, j) = mobp * offsetDeltaMatrix(0, j)
                                          + rate * paraErrors(0, j);
                offsetMatrix(0, j) += offsetDeltaMatrix(0, j);
            }//Of if
        }//Of for j

        //For the activation function.
        errorMatrix(0, i) = inputData(0, i) * (1 - inputData(0, i)) * errorSum;
    }//Of for i

    return errorMatrix;
}//Of backPropagation

/**
 * Show weights for display.
 */
void AnnLayer::showWeight()
{
    cout << weightMatrix << endl;
}//Of showWeight

/**
 * Code unit test.
 */
void AnnLayer::unitTest()
{
    AnnLayer* tempLayer = new AnnLayer(2, 3, 's', 0.01, 0.1);
    DoubleMatrix tempInput;
    tempInput.resize(1, 2);
    tempInput << 1.0, 4.0;
    printf("The input is: \r\n");
    cout << tempInput << endl;

    //printf(tempInput -> toString().data());

    //printf("The weights are: \r\n");
    cout << tempLayer -> weightMatrix << endl;

    DoubleMatrix tempOutput = tempLayer -> forward(tempInput);
    printf("The output is: \r\n");
    cout << tempOutput << endl;

    printf("Back propogation \r\n");
    tempLayer -> backPropagation(tempOutput);

    printf("Show me: \r\n");
    cout << toString() << endl;
}//Of unitTest
