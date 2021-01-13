/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "AnnLayer.h"
#include "Malloc.h"
#include "Math.h"
#include "stdio.h"

//The default constructor
AnnLayer::AnnLayer()
{
    inputSize = 1;
    outputSize = 1;
}//Of the default constructor

//Constructor for input/output size
AnnLayer::AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation)
{
    inputSize = paraInputSize;
    outputSize = paraOutputSize;

    weightMatrix = new Matrix(paraInputSize, paraOutputSize);
    offsetMatrix = new Matrix(1, paraOutputSize);

    activation = paraActivation;
}//Of the second constructor

//Destructor
AnnLayer::~AnnLayer()
{
    free(weightMatrix);
}//Of the destructor

//Convert to string for display
string AnnLayer::toString()
{
    string resultString = "I am an ANN layer with size " + to_string(inputSize)
        + "*" + to_string(outputSize) + "\r\n";
    resultString += "weight matrix: \r\n";
    resultString += weightMatrix -> toString();

    resultString += "offset matrix: \r\n";
    resultString += offsetMatrix -> toString();
    resultString += "weights ends. \r\n";

    return resultString;
}//Of toString

//Set the activation function
void AnnLayer::setActivation(char paraActivation)
{
    activation = paraActivation;
}//Of setActivation

//Activate
Matrix* AnnLayer::forward(Matrix* paraData)
{
    //printf("Forwarding, the data is: \r\n");
    //printf(paraData -> toString().data());

    printf("The weights are: \r\n");
    printf(weightMatrix -> toString().data());

    Matrix* resultData = paraData -> dot(weightMatrix);
    resultData -> addToMe(offsetMatrix);
    resultData -> activate(activation);

    printf("The resultData are: \r\n");
    printf(resultData -> toString().data());
    return resultData;
}//Of forward

//Code self test
void AnnLayer::selfTest()
{
    AnnLayer* tempLayer = new AnnLayer(2, 3, 's');
    Matrix* tempInput = new Matrix(1, 2);
    printf("The input is: \r\n");
    printf(tempInput -> toString().data());

    printf("The weights are: \r\n");
    printf(tempLayer -> weightMatrix -> toString().data());

    Matrix* tempOutput = tempLayer -> forward(tempInput);
    printf("The output is: \r\n");
    printf(tempOutput -> toString().data());
}//Of selfTest
