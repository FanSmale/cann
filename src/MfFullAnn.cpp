/**
 * The C++ Artificial Neural network project.
 * Stack a number of MfAnnLayer to form a network.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfFullAnn.h"

/**
 * The default constructor.
 */
MfFullAnn::MfFullAnn()
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
MfFullAnn::MfFullAnn(MfIntArray* paraSizes, char paraActivation, double paraRate, double paraMobp)
{
    layerSizes = paraSizes;
    activation = paraActivation;
    learningRate = paraRate;
    mobp = paraMobp;

    //Allocate space
    numLayers = layerSizes->getLength() - 1;
    layers = new MfAnnLayer* [numLayers];
    for (int i = 0; i < numLayers; i ++)
    {
        layers[i] = new MfAnnLayer(layerSizes->getValue(i), layerSizes->getValue(i + 1), activation, learningRate, mobp);
    }//Of for i

    int tempInputSize = layerSizes->getValue(0);
    int tempOutputSize = layerSizes->getValue(numLayers);
    currentInstance = new MfDoubleMatrix(1, tempInputSize);
    currentDecision = new MfDoubleMatrix(1, tempOutputSize);
    currentOutput = new MfDoubleMatrix(1, tempOutputSize);
}//Of the second constructor

/**
 * The destructor. Free space.
 */
MfFullAnn::~MfFullAnn()
{
    for(int i = 0; i < numLayers; i ++)
    {
        free(layers[i]);
    }//Of for i
    free(layers);

    free(currentInstance);
    free(currentDecision);
    free(currentOutput);
}//Of the destructor

/**
 * Convert to string for display.
 */
string MfFullAnn::toString()
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
void MfFullAnn::setActivationFunction(int paraLayer, char paraActivation)
{
    if (paraLayer >= numLayers)
    {
        throw "MfFullAnn::setActivationFunction(), layer out of range.";
    }//Of if
    layers[paraLayer]->setActivationFunction(paraActivation);
}//Of setActivationFunction

/**
 * Reset all layers.
 */
void MfFullAnn::reset()
{
    for (int i = 0; i < numLayers; i ++)
    {
        layers[i]->reset();
    }//Of for i
}//Of reset

/**
 * Forward layer by layer.
 * paraInput: A row vector for one instance.
 * Attention: We may need batch training, that is,
 *   forward a set of instances at a time.
 *   To do so, MfAnnLayer.forward() instead of this method should be revised.
 */
MfDoubleMatrix* MfFullAnn::forward(MfDoubleMatrix* paraInput)
{
    MfDoubleMatrix* tempData = paraInput;
    for (int i = 0; i < numLayers; i ++)
    {
        tempData = layers[i]->forward(tempData);
    }//Of for i
    currentOutput->cloneToMe(tempData);

    return currentOutput;
}//Of forward

/**
 * Back propagation. It relies on currentOutput computed during forward().
 *   Hence it should be invoked immediately after forward().
 * paraTarget: The actual class information.
 */
void MfFullAnn::backPropagation(MfDoubleMatrix* paraTarget)
{
    paraTarget->subtractToMe(paraTarget, currentOutput);

    currentOutput->deriveToMe(currentOutput);
    currentOutput->cwiseProductToMe(currentOutput, paraTarget);

    MfDoubleMatrix* tempLayerErrors = currentOutput;

    for (int i = numLayers - 1; i >= 0; i --)
    {
        tempLayerErrors = layers[i]->backPropagation(tempLayerErrors);
    }//Of for i
}//Of backPropagation

/**
 * Train the network with only one instance.
 * paraX: the instance (1 * m row vector).
 * paraY: the decision of the instance.
 */
void MfFullAnn::train(MfDoubleMatrix* paraX, int paraY)
{
    currentDecision->fill(0);
    currentDecision->setValue(0, paraY, 1);

    forward(paraX);
    backPropagation(currentDecision);
}//Of train

/**
 * Train the network with a dataset.
 * paraX: the data (n * m matrix).
 * paraY: the decisions (n * 1 column vector).
 */
void MfFullAnn::train(MfDoubleMatrix* paraX, MfIntArray* paraY)
{
    int tempNumInstances = paraX->getRows();
    int tempNumConditions = paraX->getColumns();
    if(tempNumInstances != paraY->getLength())
    {
        printf("The number of instances is %d while the number of labels is %d\r\n", tempNumInstances,
               paraY->getLength());
        throw "MfFullAnn::train, length not match.";
    }//Of if

    for(int i = 0; i < tempNumInstances; i ++)
    {
        //Copy this instance
        for(int j = 0; j < tempNumConditions; j ++)
        {
            currentInstance->setValue(0, j, paraX ->getValue(i, j));
        }//Of for j
        currentDecision->fill(0);
        currentDecision->setValue(0, paraY->getValue(i), 1);

        forward(currentInstance);
        backPropagation(currentDecision);
    }//Of for i
}//Of train

/**
 * Test the network with only one instance.
 * paraX: the data (n * m matrix).
 * paraY: the decision.
 * Returns: correct or not.
 */
bool MfFullAnn::test(MfDoubleMatrix* paraX, int paraY)
{
    int tempPredictionClass = -1;
    double tempMaxValue = -10;

    MfDoubleMatrix* tempPrediction = forward(paraX);
    for(int i = 0; i < tempPrediction->getColumns(); i ++)
    {
        if (tempMaxValue < tempPrediction->getValue(0, i))
        {
            tempMaxValue = tempPrediction->getValue(0, i);
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
double MfFullAnn::test(MfDoubleMatrix* paraX, MfIntArray* paraY)
{
    int tempNumInstances = paraX->getRows();
    int tempNumConditions = paraX->getColumns();
    int tempPredictionClass;
    double tempMaxValue;

    if(tempNumInstances != paraY->getLength())
    {
        throw "MfFullAnn::test, length not match";
    }//Of if

    //Initialize this member variable
    numCorrect = 0;
    for(int i = 0; i < tempNumInstances; i ++)
    {
        tempPredictionClass = 0;
        tempMaxValue = -10;

        //Copy this instance
        for(int j = 0; j < tempNumConditions; j ++)
        {
            currentInstance->setValue(0, j, paraX->getValue(i, j));
        }//Of for j
        printf("The %dth testing data is: ", i);
        cout << currentInstance->toString() << ", " << paraY->getValue(i) << endl;

        currentOutput = forward(currentInstance);
        for(int j = 0; j < currentOutput->getColumns(); j ++)
        {
            if (tempMaxValue < currentOutput->getValue(0, j))
            {
                tempMaxValue = currentOutput->getValue(0, j);
                tempPredictionClass = j;
            }//Of if
        }//Of for j
        printf("The predicted class is %d\r\n", tempPredictionClass);

        if (tempPredictionClass == paraY->getValue(i))
        {
            numCorrect ++;
        }//Of if
    }//Of for i
    return (numCorrect + 0.0) / tempNumInstances;
}//Of test

/**
 * Show weight. For debug only.
 */
void MfFullAnn::showWeight()
{
    printf("numLayers = %d \r\n", numLayers);
    for(int i = 0; i < numLayers; i ++)
    {
        layers[i]->showWeight();
    }//Of for i
}//Of showWeight

/**
 * Code unit test.
 */
void MfFullAnn::unitTest()
{
    printf("Test MfFullAnn.cpp\r\n");
    int tempArray[3] = {3, 5, 7};
    MfIntArray* tempMfIntArray = new MfIntArray(3);
    for(int i = 0; i < 3; i ++)
    {
        tempMfIntArray->setValue(i, tempArray[i]);
    }//Of for i

    printf("MfIntArray constructed. \r\n");
    MfFullAnn* tempMfFullAnn = new MfFullAnn(tempMfIntArray, 's', 0.01, 0.1);

    printf("MfFullAnn built\r\n");

    MfDoubleMatrix* tempData = new MfDoubleMatrix(1, 3);
    tempData->setValue(0, 0, 1.2);
    tempData->setValue(0, 1, 1.6);
    tempData->setValue(0, 2, 2.7);
    printf("Input data built\r\n");

    tempData = tempMfFullAnn->forward(tempData);
    printf("After forward \r\n");

    cout << tempData->toString() <<endl;

    printf("Back propagation:\r\n");
    tempMfFullAnn->backPropagation(tempData);

    //Build the network structure
    /*
    MfIntArray tempMfIntArray;
    tempMfIntArray.resize(1, 4);
    tempMfIntArray(0) = 4;
    tempMfIntArray(1) = 5;
    tempMfIntArray(2) = 5;
    tempMfIntArray(3) = 3;
    MfFullAnn* tempMfFullAnn = new MfFullAnn(tempMfIntArray, 's', 0.1, 0.1);

    printf("Train:\r\n");
    */

    /*
    MfDoubleMatrix tempX;
    tempX.resize(3, 2);
    tempX(0, 0) = 1.0;
    tempX(0, 1) = 2.0;
    tempX(1, 0) = 3.0;
    tempX(1, 1) = 4.0;
    tempX(2, 0) = 1.0;
    tempX(2, 1) = 4.0;

    MfIntArray tempY;
    tempY.resize(1, 3);
    tempY(0, 0) = 0;
    tempY(0, 1) = 0;
    tempY(0, 2) = 1;
    */

}//Of selfTest

/**
 * Training/testing test.
 */
void MfFullAnn::trainingTestingTest()
{
    string tempString = "D:\\C\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();

    MfDataReader* tempReader = new MfDataReader(tempFilename);
    tempReader->randomize();
    tempReader->splitInTwo(0.6);

    MfDoubleMatrix* tempX = tempReader->getTrainingX();
    MfIntArray* tempY = tempReader->getTrainingY();
    MfDoubleMatrix* tempTestingX = tempReader->getTestingX();
    MfIntArray* tempTestingY = tempReader->getTestingY();

    printf("Training/testing data generated:\r\n");

    int tempDepth = 4;
    int tempArray[tempDepth] = {4, 5, 6, 3};
    MfIntArray* tempMfIntArray = new MfIntArray(tempDepth);
    for(int i = 0; i < tempDepth; i ++)
    {
        tempMfIntArray->setValue(i, tempArray[i]);
    }//Of for i
    MfFullAnn* tempMfFullAnn = new MfFullAnn(tempMfIntArray, 's', 0.1, 0.1);

    printf("Ann constructed:\r\n");
    for(int i = 0; i < 1000; i ++)
    {
        //printf("i = %d:\r\n", i);
        tempMfFullAnn->train(tempX, tempY);
        if (i % 200 == 0)
        {
           tempMfFullAnn->showWeight();
        }//Of if
    }//Of for i

    printf("After training:\r\n\r\n\r\n\r\n");
    double tempPrecision = tempMfFullAnn->test(tempTestingX, tempTestingY);
    printf("After testing, the precision is %lf:\r\n", tempPrecision);

    free(tempMfIntArray);
    free(tempMfFullAnn);

    printf("Finish. \r\n");
}//Of trainingTestingTest

/**
 * Cross validation test.
 */
void MfFullAnn::crossValidationTest()
{
    string tempString = "D:\\C\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();

    MfDataReader* tempReader = new MfDataReader(tempFilename);
    tempReader->randomize();

    int tempDepth = 4;
    int tempArray[tempDepth] = {4, 8, 8, 3};
    MfIntArray* tempMfIntArray = new MfIntArray(tempDepth);
    for(int i = 0; i < tempDepth; i ++)
    {
        tempMfIntArray->setValue(i, tempArray[i]);
    }//Of for i

    int tempNumFolds = 5;
    int tempCorrectSum = 0;

    //Attention: 这里声明即赋值, 后面的多个网络结果才正确.
    // 先声明为空指针, 后面再赋值, 都不行.
    MfFullAnn* tempMfFullAnn = new MfFullAnn(tempMfIntArray, 's', 0.1, 0.1);
    for(int i = 0; i < tempNumFolds; i++)
    {
        //消除上个网络的权值影响.
        tempMfFullAnn->reset();
        tempReader->crossValidationSplit(tempNumFolds, i);

        MfDoubleMatrix* tempX = tempReader->getTrainingX();
        MfIntArray* tempY = tempReader->getTrainingY();
        MfDoubleMatrix* tempTestingX = tempReader->getTestingX();
        MfIntArray* tempTestingY = tempReader->getTestingY();

        printf("Training/testing data generated:\r\n");

        for(int i = 0; i < 1000; i ++)
        {
            tempMfFullAnn->train(tempX, tempY);
            //if (i % 300 == 0)
            //{
            //   tempMfFullAnn->showWeight();
            //}//Of if
        }//Of for i

        printf("After training:\r\n\r\n\r\n\r\n");

        double tempPrecision = tempMfFullAnn->test(tempTestingX, tempTestingY);
        printf("After testing, the precision is %lf:\r\n", tempPrecision);
        tempCorrectSum += tempMfFullAnn->getNumCorrect();
    }//Of for i
    free(tempMfFullAnn);

    printf("Total correct: %d. \r\n", tempCorrectSum);
    printf("Finish. \r\n");
}//Of crossValidationTest

