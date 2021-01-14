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

//The default constructor
AnnLayer::AnnLayer()
{
    inputSize = 1;
    outputSize = 1;
}//Of the default constructor

//Constructor for input/output size
AnnLayer::AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation)
{
    printf("AnnLayer constructor\r\n");
    inputSize = paraInputSize;
    outputSize = paraOutputSize;

    printf("Trying to resize the weight matrix with %d and %d.\r\n", paraInputSize, paraOutputSize);
    weightMatrix.resize(paraInputSize, paraOutputSize);
    //weightMatrix.setRandom();
    weightMatrix.setOnes();
    printf("Matrix resized \r\n");
    offsetMatrix.resize(1, paraOutputSize);
    offsetMatrix.setOnes();

    activation = paraActivation;
}//Of the second constructor

//Destructor
AnnLayer::~AnnLayer()
{
    //free(weightMatrix);
}//Of the destructor

//Convert to string for display
string AnnLayer::toString()
{
    string resultString = "I am an ANN layer with size " + to_string(inputSize)
                          + "*" + to_string(outputSize) + "\r\n";
    //resultString += "weight matrix: \r\n";
    //resultString += weightMatrix -> toString();

    //resultString += "offset matrix: \r\n";
    //resultString += offsetMatrix -> toString();
    //resultString += "weights ends. \r\n";

    return resultString;
}//Of toString

//Set the activation function
void AnnLayer::setActivation(char paraActivation)
{
    activation = paraActivation;
}//Of setActivation

//Activate for the given value, independent of this object
double AnnLayer::activate(double paraValue, char paraFunction)
{
    switch (paraFunction)
    {
    case 's':
        return 1 / (1 + exp(-paraValue));
    case 'r':
        if (paraValue > 0)
        {
            return paraValue;
        }
        else
        {
            return 0;
        }//Of if
    default:
        return paraValue;
    }//Of switch
}//Of activate

//Activate
DoubleMatrix AnnLayer::forward(DoubleMatrix paraData)
{
    //printf("Forwarding, the data is: \r\n");
    //printf(paraData -> toString().data());

    //printf("The weights are: \r\n");
    //printf(weightMatrix -> toString().data());

    printf("paraData: \r\n");
    cout << paraData << endl;
    printf("weights: \r\n");
    cout << weightMatrix << endl;
    DoubleMatrix resultData = paraData * weightMatrix;
    printf("After weighted sum: \r\n");
    cout << resultData << endl;
    resultData += offsetMatrix;
    printf("After adding offset: \r\n");
    cout << resultData << endl;
    //resultData -> activate(activation);
    printf("before activiate:\r\n");
    //cout << resultData(1, 0) << endl;
    cout << resultData.data() << endl;

    resultData(1, 0) = activate(resultData(1, 0), activation);
    printf("After activiate: \r\n");

    return resultData;
}//Of forward

//Code self test
void AnnLayer::selfTest()
{
    AnnLayer* tempLayer = new AnnLayer(2, 3, 's');
    Eigen::Matrix<double, 1, 2> tempInput;
    tempInput << 1.0, 4.0;
    printf("The input is: \r\n");
    //printf(tempInput -> toString().data());

    printf("The weights are: \r\n");
    //printf(tempLayer -> weightMatrix -> toString().data());

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tempOutput = tempLayer -> forward(tempInput);
    printf("The output is: \r\n");
    //printf(tempOutput -> toString().data());
}//Of selfTest
