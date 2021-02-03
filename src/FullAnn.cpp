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
#include "EigenSupport.h"
#include "FullAnn.h"
#include "DataReader.h"

/**
 * The default constructor.
 */
FullAnn::FullAnn()
{
    numLayers = 0;
    layers = nullptr;
}//Of the default constructor

/**
 * The second constructor.
 * paraSizes: The sizes of layers, the first element is the number of conditions,
 *   and the last element is the number of classes.
 *   e.g., {4, 6, 10, 3} for iris.
 * paraActivation: The activation function name, e.g., 's' for sigmoid.
 *   This parameter and the following two will be stored as member variables.
 * paraRate: The learning rate.
 * paraMobp: The mobp.
 */
FullAnn::FullAnn(IntArray paraSizes, char paraActivation, double paraRate, double paraMobp)
{
    layerSizes = paraSizes;
    activation = paraActivation;
    rate = paraRate;
    mobp = paraMobp;

    //Allocate space
    numLayers = layerSizes.cols() - 1;
    layers = new AnnLayer* [numLayers];

    for (int i = 0; i < numLayers; i ++)
    {
        layers[i] = new AnnLayer(layerSizes(i), layerSizes(i + 1),
                                 activation, rate, mobp);
    }//Of for i
}//Of the second constructor

/**
 * The destructor. Free space.
 */
FullAnn::~FullAnn()
{
    for(int i = 0; i < numLayers; i ++)
    {
        free(layers[i]);
    }//Of for i
    free(layers);
}//Of the destructor

/**
 * Convert to string for display.
 */
string FullAnn::toString()
{
    string resultString = "I am a full ANN with " + to_string(numLayers)
                          + " Layers.\r\n";

    return resultString;
}//Of toString

/**
 * Set activation function for the given layer.
 *   Different layers can have different activation functions.
 * paraLayer: The layer index.
 * paraActivation: The new activation function.
 */
void FullAnn::setActivationFunction(int paraLayer, char paraActivation)
{
    if (paraLayer >= numLayers)
    {
        throw OUT_OF_RANGE_EXCEPTION;
    }//Of if
    layers[paraLayer] -> setActivationFunction(paraActivation);
}//Of setActivationFunction

/**
 * Setter.
 */
void FullAnn::setRate(double paraRate)
{
    rate = paraRate;
}//Of setRate

/**
 * Setter.
 */
void FullAnn::setMobp(double paraMobp)
{
    mobp = paraMobp;
}//Of setMobp

/**
 * Reset all layers.
 */
void FullAnn::reset()
{
    for (int i = 0; i < numLayers; i ++)
    {
        layers[i] -> reset();
    }//Of for i
}//Of reset

/**
 * Forward layer by layer.
 * paraInput: A row vector for one instance.
 * Attention: We may need batch training, that is,
 *   forward a set of instances at a time.
 *   To do so, AnnLayer.forward() instead of this method should be revised.
 */
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

/**
 * Back propagation. It relies on currentOutput computed during forward().
 *   Hence it should be invoked immediately after forward().
 * paraTarget: The actual class information.
 */
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

/**
 * Train the network with only one instance.
 * paraX: the instance (1 * m row vector).
 * paraY: the decision of the instance.
 * paraNumClasses: the number of classes of this dataset.
 */
void FullAnn::train(DoubleMatrix paraX, int paraY, int paraNumClasses)
{
    DoubleMatrix tempDecision;
    tempDecision.resize(1, paraNumClasses);
    tempDecision.fill(0);
    tempDecision(0, paraY) = 1;

    forward(paraX);
    backPropagation(tempDecision);
}//Of train

/**
 * Train the network with a dataset.
 * paraX: the data (n * m matrix).
 * paraY: the decisions (n * 1 column vector).
 * paraNumClasses: the number of classes of this dataset.
 */
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
            tempData(0, j) = paraX(i, j);
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

/**
 * Test the network with only one instance.
 * paraX: the data (n * m matrix).
 * paraY: the decision.
 * Returns: correct or not.
 */
bool FullAnn::test(DoubleMatrix paraX, int paraY)
{
    int tempPredictionClass = -1;
    double tempMaxValue = -10;

    DoubleMatrix tempPrediction = forward(paraX);
    for(int i = 0; i < tempPrediction.cols(); i ++)
    {
        if (tempMaxValue < tempPrediction(i))
        {
            tempMaxValue = tempPrediction(i);
            tempPredictionClass = i;
        }//Of if
    }//Of for i
    printf("The predicted class is %d\r\n", tempPredictionClass);

    return tempPredictionClass == paraY;
}//Of test

/**
 * Test the network with a dataset.
 * paraX: the data (n * m matrix).
 * paraY: the decisions (n * 1 column vector).
 * Returns: the prediction accuracy.
 */
double FullAnn::test(DoubleMatrix paraX, IntArray paraY)
{
    int tempNumInstances = paraX.rows();
    int tempNumConditions = paraX.cols();
    int tempPredictionClass;
    double tempMaxValue;

    if(tempNumInstances != paraY.cols())
    {
        throw LENGTH_NOT_MATCH_EXCEPTION;
    }//Of if

    DoubleMatrix tempData;
    tempData.resize(1, tempNumConditions);

    //Initialize this member variable
    numCorrect = 0;
    DoubleMatrix tempPrediction;
    for(int i = 0; i < tempNumInstances; i ++)
    {
        tempPredictionClass = 0;
        tempMaxValue = -10;

        //Copy this instance
        for(int j = 0; j < tempNumConditions; j ++)
        {
            tempData(0, j) = paraX(i, j);
        }//Of for j
        printf("The %dth testing data is: ", i);
        cout << tempData << ", " << paraY(0, i) << endl;

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
            numCorrect ++;
        }//Of if
    }//Of for i
    return (numCorrect + 0.0) / tempNumInstances;
}//Of test

/**
 * Getter.
 */
int FullAnn::getNumCorrect()
{
    return numCorrect;
}//Of getNumCorrect

/**
 * Show weight. For debug only.
 */
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

/**
 * Code unit test.
 */
void FullAnn::unitTest()
{
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

    //Build the network structure
    /*
    IntArray tempIntArray;
    tempIntArray.resize(1, 4);
    tempIntArray(0) = 4;
    tempIntArray(1) = 5;
    tempIntArray(2) = 5;
    tempIntArray(3) = 3;
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's', 0.1, 0.1);

    printf("Train:\r\n");
    */

    /*
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
    */

}//Of selfTest

/**
 * Training/testing test.
 */
void FullAnn::trainingTestingTest()
{
    string tempString = "D:\\C\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();

    DataReader tempReader(tempFilename);
    tempReader.randomize();
    tempReader.splitInTwo(0.6);

    DoubleMatrix tempX = tempReader.getTrainingX()[0];
    IntArray tempY = tempReader.getTrainingY()[0];
    DoubleMatrix tempTestingX = tempReader.getTestingX()[0];
    IntArray tempTestingY = tempReader.getTestingY()[0];

    printf("Training/testing data generated:\r\n");

    int tempDepth = 4;
    int tempArray[tempDepth] = {4, 5, 6, 3};
    IntArray tempIntArray;
    tempIntArray.resize(1, tempDepth);
    for(int i = 0; i < tempDepth; i ++)
    {
        tempIntArray(i) = tempArray[i];
    }//Of for i
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's', 0.1, 0.1);

    printf("Ann constructed:\r\n");
    for(int i = 0; i < 1000; i ++)
    {
        tempFullAnn -> train(tempX, tempY, 3);
        if (i % 200 == 0)
        {
           tempFullAnn -> showWeight();
        }//Of if
    }//Of for i

    printf("After training:\r\n\r\n\r\n\r\n");

    //double tempPrecision = tempFullAnn -> test(tempTestingX, tempTestingY);
    double tempPrecision = tempFullAnn -> test(tempX, tempY);
    printf("After testing, the precision is %lf:\r\n", tempPrecision);

    printf("Finish. \r\n");
}//Of trainingTestingTest

/**
 * Cross validation test.
 */
void FullAnn::crossValidationTest()
{
    string tempString = "D:\\C\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();

    DataReader tempReader(tempFilename);

    tempReader.randomize();

    int tempDepth = 4;
    int tempArray[tempDepth] = {4, 8, 8, 3};
    IntArray tempIntArray;
    tempIntArray.resize(1, tempDepth);
    for(int i = 0; i < tempDepth; i ++)
    {
        tempIntArray(i) = tempArray[i];
    }//Of for i

    int tempNumFolds = 4;
    int tempCorrectSum = 0;

    //Attention: 这里声明即赋值, 后面的多个网络结果才正确.
    // 先声明为空指针, 后面再赋值, 都不行.
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's', 0.1, 0.1);
    for(int i = 0; i < tempNumFolds; i++)
    {
        //Attention: 在这里生成新的网络, 第2个网络及以后均只能给出一个预测值.
        //  原因未知.
        //free(tempFullAnn);
        //tempFullAnn = new FullAnn(tempIntArray, 's', 0.1, 0.1);

        //消除上个网络的权值影响.
        tempFullAnn -> reset();
        tempReader.crossValidationSplit(tempNumFolds, i);

        DoubleMatrix tempX = tempReader.getTrainingX()[0];
        IntArray tempY = tempReader.getTrainingY()[0];
        DoubleMatrix tempTestingX = tempReader.getTestingX()[0];
        IntArray tempTestingY = tempReader.getTestingY()[0];

        printf("Training/testing data generated:\r\n");

        for(int i = 0; i < 1000; i ++)
        {
            tempFullAnn -> train(tempX, tempY, 3);
            //if (i % 300 == 0)
            //{
            //   tempFullAnn -> showWeight();
            //}//Of if
        }//Of for i

        printf("After training:\r\n\r\n\r\n\r\n");

        double tempPrecision = tempFullAnn -> test(tempTestingX, tempTestingY);
        printf("After testing, the precision is %lf:\r\n", tempPrecision);
        tempCorrectSum += tempFullAnn -> getNumCorrect();
    }//Of for i
    free(tempFullAnn);

    printf("Total correct: %d. \r\n", tempCorrectSum);
    printf("Finish. \r\n");
}//Of crossValidationTest

