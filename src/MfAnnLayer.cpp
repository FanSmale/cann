/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfAnnLayer.h"
#include "Activator.h"

/**
 * The default constructor
 */
MfAnnLayer::MfAnnLayer()
{
    inputSize = 1;
    outputSize = 1;
    activator = nullptr;
}//Of the default constructor

/**
 * Constructor for input/output size
 */
MfAnnLayer::MfAnnLayer(int paraInputSize, int paraOutputSize, char paraActivation,
                   double paraRate, double paraMobp)
{
    //printf("MfAnnLayer constructor\r\n");
    inputSize = paraInputSize;
    outputSize = paraOutputSize;
    learningRate = paraRate;
    mobp = paraMobp;

    weightMatrix = new MfDoubleMatrix(paraInputSize, paraOutputSize);
    weightDeltaMatrix = new MfDoubleMatrix(paraInputSize, paraOutputSize);
    errorMatrix = new MfDoubleMatrix(1, paraInputSize);
    inputData = new MfDoubleMatrix(1, paraInputSize);;
    outputData = new MfDoubleMatrix(1, paraOutputSize);

    for(int i = 0; i < paraInputSize; i ++)
    {
        for (int j = 0; j < paraOutputSize; j ++)
        {
            weightMatrix->setValue(i, j, (double)rand()/RAND_MAX);
        }//Of for j
    }//Of for i

    //weightMatrix.setOnes();
    offsetMatrix = new MfDoubleMatrix(1, paraOutputSize);
    offsetDeltaMatrix = new MfDoubleMatrix(1, paraOutputSize);
    for (int i = 0; i < paraOutputSize; i ++)
    {
        offsetMatrix->setValue(0, i, (double)rand()/RAND_MAX);
    }//Of for i

    activator = new Activator(paraActivation);

    outputData->setActivator(activator);
}//Of the second constructor

/**
 * Destructor.
 */
MfAnnLayer::~MfAnnLayer()
{
    free(activator);
    free(weightMatrix);
    free(weightDeltaMatrix);
    free(errorMatrix);
    free(inputData);
    free(outputData);
    free(offsetMatrix);
    free(offsetDeltaMatrix);
}//Of the destructor

/**
 * Convert to string for display.
 * Returns: The string showing myself.
 */
string MfAnnLayer::toString()
{
    string resultString = "I am an ANN layer with size " + to_string(inputSize)
                          + "*" + to_string(outputSize) + "\r\n";
    resultString += "weight matrix: \r\n";
    resultString += weightMatrix->toString();

    return resultString;
}//Of toString

/**
 * Show weights for display.
 */
void MfAnnLayer::showWeight()
{
    cout << weightMatrix->toString() << endl;
}//Of showWeight

/**
 * Set the activation function.
 * paraFunction: the activation function in char.
 */
void MfAnnLayer::setActivationFunction(char paraFunction)
{
    activator->setActivationFunction(paraFunction);
}//Of setActivationFunction

/**
 * Getter.
 */
Activator* MfAnnLayer::getActivator()
{
    return activator;
}//Of getActivator

/**
 * Reset weights and other variables.
 */
 void MfAnnLayer::reset()
 {
    for(int i = 0; i < inputSize; i ++)
    {
        for (int j = 0; j < outputSize; j ++)
        {
            weightMatrix->setValue(i, j, (double)rand()/RAND_MAX);
        }//Of for j
    }//Of for i

    for (int i = 0; i < outputSize; i ++)
    {
        offsetMatrix->setValue(0, i, (double)rand()/RAND_MAX);
    }//Of for i
 }//Of reset

/**
 * Getter.
 */
int MfAnnLayer::getInputSize()
{
    return inputSize;
}//Of getInputSize

/**
 * Getter.
 */
int MfAnnLayer::getOutputSize()
{
    return outputSize;
}//Of getOutputSize

/**
 * Forward computing. No new space is allocated in this process.
 * paraData: now only one instance is supported.
 */
MfDoubleMatrix* MfAnnLayer::forward(MfDoubleMatrix* paraData)
{
    //printf("Forwarding, the data is: \r\n");
    //cout << paraData->toString() << endl;

    //printf("The weights are: \r\n");
    //printf(weightMatrix->toString().data());

    inputData->cloneToMe(paraData);

    outputData->timesToMe(inputData, weightMatrix);
    outputData->addToMe(outputData, offsetMatrix);

    outputData->activate();

    return outputData;
}//Of forward

/**
 * Back propagation.
 *   The activation function of the last layer is considered.
 * paraErrors: the error at the output end.
 * Returns: the error at the input end (attention: activation function).
 */
MfDoubleMatrix* MfAnnLayer::backPropagation(MfDoubleMatrix* paraErrors)
{
    //printf("MfAnnLayer::backPropagation test 1\r\n");
    double tempValue1, tempValue2, tempValue3, tempValue4;
    double tempErrorSum;
    for(int i = 0; i < inputSize; i ++)
    {
        tempErrorSum = 0;

        //Weights adjusting
        for(int j = 0; j < outputSize; j ++)
        {
            //printf("MfAnnLayer::backPropagation test 2.1.1, j = %d\r\n", j);
            tempErrorSum += paraErrors->getValue(0, j) * weightMatrix->getValue(i, j);
            tempValue1 = mobp * weightDeltaMatrix->getValue(i, j)
                + learningRate * paraErrors->getValue(0, j) * inputData->getValue(0, i);
            weightDeltaMatrix->setValue(i, j, tempValue1);

            tempValue2 = weightMatrix->getValue(i, j);
            weightMatrix->setValue(i, j, tempValue1 + tempValue2);

            if (i == inputSize - 1)
            {
                // Offset adjusting
                tempValue1 = offsetDeltaMatrix->getValue(0, j);
                tempValue2 = paraErrors->getValue(0, j);
                tempValue3 = mobp * tempValue1 + learningRate * tempValue2;
                offsetDeltaMatrix->setValue(0, j, tempValue3);
                tempValue4 = offsetMatrix->getValue(0, j);
                //offsetMatrix(0, j) += offsetDeltaMatrix(0, j);
                offsetMatrix->setValue(0, j, tempValue4 + tempValue3);
            }//Of if
        }//Of for j

        //For the activation function.
        tempValue1 = inputData->getValue(0, i);
        //errorMatrix->setValue(0, i, tempValue1 * (1 - tempValue1) * tempErrorSum);
        errorMatrix->setValue(0, i, activator->derive(tempValue1) * tempErrorSum);
    }//Of for i

    return errorMatrix;
}//Of backPropagation

/**
 * Code unit test.
 */
void MfAnnLayer::unitTest()
{
    MfAnnLayer* tempLayer = new MfAnnLayer(2, 3, 's', 0.01, 0.1);
    MfDoubleMatrix* tempInput = new MfDoubleMatrix(1, 2);
    tempInput->setValue(0, 0, 1.0);
    tempInput->setValue(0, 1, 4.0);
    //tempInput << 1.0, 4.0;
    printf("The input is: \r\n");
    cout << tempInput->toString() << endl;

    printf("The weights are: \r\n");
    cout << tempLayer->weightMatrix->toString() << endl;

    MfDoubleMatrix* tempOutput = tempLayer->forward(tempInput);
    printf("The output is:\r\n");
    cout << tempOutput->toString() << endl;

    printf("Back propogation \r\n");
    tempLayer->backPropagation(tempOutput);

    printf("Show me: \r\n");
    cout << tempLayer->toString() << endl;
}//Of unitTest
