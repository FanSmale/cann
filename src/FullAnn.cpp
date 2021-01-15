/*
 * The C++ Artificial Neural network project.
 * Stack a number of AnnLayer to for a
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <iostream>
#include "MfMath.h"
#include "FullAnn.h"

//The default constructor
FullAnn::FullAnn()
{
    numLayers = 0;
    layers = nullptr;
}//Of the default constructor

//The second constructor
FullAnn::FullAnn(IntArray paraSizes, char paraActivation, double paraRate, double paraMobp)
{
    layerSizes = paraSizes;
    activation = paraActivation;
    rate = paraRate;
    mobp = paraMobp;

    //printf("Test the constructor of FullAnn.cpp\r\n");
    //Allocate space
    numLayers = layerSizes.cols() - 1;
    layers = new AnnLayer* [numLayers];
    //printf("layers allocated \r\n");
    for (int i = 0; i < numLayers; i ++)
    {
        //printf("layer %d \r\n", i);
        layers[i] = new AnnLayer(layerSizes(i), layerSizes(i + 1),
                                 activation, rate, mobp);
    }//Of for i
    printf("End of the constructor of FullAnn.cpp\r\n");
}//Of the second constructor

//Destructor
FullAnn::~FullAnn()
{
    for(int i = 0; i < numLayers; i ++)
    {
        free(layers[i]);
    }//Of for i
    free(layers);
}//Of the destructor

//Convert to string for display
string FullAnn::toString()
{
    string resultString = "I am a full ANN with " + to_string(numLayers)
                          + " Layers.\r\n";

    return resultString;
}//Of toString

void FullAnn::setActivation(int paraLayer, char paraActivation)
{
    if (paraLayer >= numLayers)
    {
        throw OUT_OF_RANGE_EXCEPTION;
    }//Of if
    layers[paraLayer] -> setActivation(paraActivation);
}//Of setActivation

//Forward layer by layer
DoubleMatrix FullAnn::forward(DoubleMatrix paraInput)
{
    DoubleMatrix tempData = paraInput;
    for (int i = 0; i < numLayers; i ++)
    {
        tempData = layers[i] -> forward(tempData);
        //printf("After layer %d.\r\n", i);
        //cout << tempData <<endl;
        //printf(tempData -> toString().data());
    }//Of for i

    currentOutput = tempData;
    return currentOutput;
}//Of forward

//Back propagation
void FullAnn::backPropagation(DoubleMatrix paraTarget)
{
    int tempSize = layers[numLayers - 1] -> outputSize;
    MatrixXd tempOnes = MatrixXd::Ones(1, tempSize);

    DoubleMatrix layerErrors = currentOutput.cwiseProduct(tempOnes - currentOutput).cwiseProduct(paraTarget - currentOutput);
    //printf("backPropagation, layerErrors = \r\n");
    //printf("numLayers = %d \r\n", numLayers);
    //cout << layerErrors << endl;

    for (int i = numLayers - 1; i >= 0; i --)
    {
        //printf("Layer %d.\r\n", i);
        layerErrors = layers[i] -> backPropagation(layerErrors);
        //cout << layerErrors <<endl;
    }//Of for i
}//Of backPropagation

//Train the network
void FullAnn::train(DoubleMatrix paraX, IntArray paraY, int paraNumClasses)
{
    int tempNumInstances = paraX.rows();
    int tempNumConditions = paraX.cols();
    if(tempNumInstances != paraY.cols())
    {
        throw LENGTH_NOT_MATCH_EXCEPTION;
    }//Of if

    DoubleMatrix tempData;
    tempData.resize(1, tempNumConditions);
    DoubleMatrix tempDecision;
    tempDecision.resize(1, paraNumClasses);
    //printf("The data has %d instances and %d conditions.\r\n", tempNumInstances, tempNumConditions);
    for(int i = 0; i < tempNumInstances; i ++)
    {
        //Copy this instance
        for(int j = 0; j < tempNumConditions; j ++)
        {
            tempData(j) = paraX(i, j);
        }//Of for j
        tempDecision.fill(0);
        tempDecision(0, paraY(0, i)) = 1;

        //printf("The %dth training data is: ", i);
        //cout << tempData << endl;
        //printf("The class information is: ");
        //cout << tempDecision << endl;

        forward(tempData);
        backPropagation(tempDecision);
    }//Of for i
}//Of train

//Test the network
double FullAnn::test(DoubleMatrix paraX, IntArray paraY)
{
    int tempNumInstances = paraX.rows();
    int tempNumConditions = paraX.cols();
    double tempCorrect = 0;
    int tempPredictionClass;
    double tempMaxValue;

    if(tempNumInstances != paraY.cols())
    {
        throw LENGTH_NOT_MATCH_EXCEPTION;
    }//Of if

    DoubleMatrix tempData;
    tempData.resize(1, tempNumConditions);

    DoubleMatrix tempPrediction;
    for(int i = 0; i < tempNumInstances; i ++)
    {
        tempPredictionClass = 0;
        tempMaxValue = -10;

        //Copy this instance
        for(int j = 0; j < tempNumConditions; j ++)
        {
            tempData(j) = paraX(i, j);
        }//Of for j
        printf("The %dth testing data is: ", i);
        cout << tempData << endl;

        tempPrediction = forward(tempData);
        for(int j = 0; j < tempPrediction.cols(); j ++)
        {
            if (tempMaxValue < tempPrediction(j))
            {
                tempMaxValue = tempPrediction(j);
                tempPredictionClass = j;
            }//Of if
        }//Of for j
        printf("The predicted class is %d\r\n", tempPredictionClass);

        if (tempPredictionClass == paraY(0, i))
        {
            tempCorrect ++;
        }//Of if
    }//Of for i
    return tempCorrect/tempNumInstances;
}//Of test

//Show weight
void FullAnn::showWeight()
{
    //printf("The weights are:\r\n");
    printf("numLayers = %d \r\n", numLayers);
    for(int i = 0; i < numLayers; i ++)
    {
        layers[i] -> showWeight();
    }//Of for i
    //printf("showWeight end:\r\n");
}//Of showWeight

//Code self test
void FullAnn::selfTest()
{
    /*
    printf("Test FullAnn.cpp\r\n");
    int tempArray[3] = {3, 5, 7};
    IntArray tempIntArray;
    tempIntArray.resize(1, 3);
    for(int i = 0; i < 3; i ++)
    {
        tempIntArray(i) = tempArray[i];
    }//Of for i

    printf("IntArray constructed. \r\n");
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's', 0.01, 0.1);

    printf("FullAnn built\r\n");

    DoubleMatrix tempData;
    tempData.resize(1, 3);
    tempData << 1.2, 1.6, 2.7;
    printf("Input data built\r\n");

    tempData = tempFullAnn -> forward(tempData);
    printf("After forward \r\n");

    cout << tempData <<endl;

    printf("Back propagation:\r\n");
    tempFullAnn -> backPropagation(tempData);
    */

    //Build the network structure
    IntArray tempIntArray;
    tempIntArray.resize(1, 3);
    tempIntArray(0) = 2;
    tempIntArray(1) = 4;
    tempIntArray(2) = 2;
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's', 0.1, 0.1);

    printf("Train:\r\n");

    DoubleMatrix tempX;
    tempX.resize(3, 2);
    tempX(0, 0) = 1.0;
    tempX(0, 1) = 2.0;
    tempX(1, 0) = 3.0;
    tempX(1, 1) = 4.0;
    tempX(2, 0) = 1.0;
    tempX(2, 1) = 4.0;

    IntArray tempY;
    tempY.resize(1, 3);
    tempY(0, 0) = 0;
    tempY(0, 1) = 0;
    tempY(0, 2) = 1;

    //printf("Data generated:\r\n");
    for(int i = 0; i < 1000; i ++)
    {
        tempFullAnn -> train(tempX, tempY, 2);
        if (i % 10 == 0)
        {
            tempFullAnn -> showWeight();
        }//Of if
    }//Of for i

    printf("After training:\r\n\r\n\r\n\r\n");

    double tempPrecision = tempFullAnn -> test(tempX, tempY);
    printf("After testing, the precision is %lf:\r\n", tempPrecision);

    printf("Finish. \r\n");

    /*
    MfMatrix* tempData = new MfMatrix(1, 3);
    tempData -> setValue(0, 0, 3.0);
    tempData -> setValue(0, 1, 2.0);
    tempData -> setValue(0, 2, 1.0);

    for (int i = 0; i < tempFullAnn -> numLayers; i ++)
    {
        tempData = tempFullAnn -> layers[i] -> forward(tempData);
        printf("After layer %d.\r\n", i);
        printf(tempData -> toString().data());
    }//Of for i

    printf("The final results are: ");
    printf(tempData -> toString().data());
    */

}//Of selfTest
