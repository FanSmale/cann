/*
 * The C++ Artificial Neural network project.
 * This class handles CNN layer.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfRnnLayer.h"

/**
 * The empty constructor.
 */
MfRnnLayer::MfRnnLayer()
{
    inputSize = 50;
    hiddenSize = 100;
    outputSize = 50;
    learningRate = 0.1;
}//Of the empty constructor

/**
 * The second constructor.
 */
MfRnnLayer::MfRnnLayer(int paraInputSize, int paraHiddenSize, int paraOutputSize,
                       int paraMaxSequenceLength, double paraRate)
{
    inputSize = paraInputSize;
    hiddenSize = paraHiddenSize;
    outputSize = paraOutputSize;
    maxSequenceLength = paraMaxSequenceLength;
    learningRate = paraRate;

    activator = new Activator('s');
}//Of the second constructor

/**
 * The destructor.
 */
MfRnnLayer::~MfRnnLayer()
{

}//Of the destructor

/**
 * Initialize the net with random weights..
 */
void MfRnnLayer::initialize()
{
    if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0)
    {
        printf("MfRnnLayer::initialize, Illegal layer sizes.\r\n");
        throw "Illegal layer sizes.";
    }//Of if

    //Step 1. Create weight matrices
    double tempScale = 0.1;

    Wxh = new MfDoubleMatrix(hiddenSize, inputSize);
    Wxh->fill(0, tempScale);

    Whh = new MfDoubleMatrix(hiddenSize, hiddenSize);
    Whh->fill(0, tempScale);

    Why = new MfDoubleMatrix(outputSize, hiddenSize);
    Why->fill(0, tempScale);

    bh = new MfDoubleMatrix(1, hiddenSize);
    bh->fill(0);

    by = new MfDoubleMatrix(1, outputSize);
    by->fill(0);

    h = new MfDoubleMatrix(1, hiddenSize);

    ////// Training state //////

    gWxh = new MfDoubleMatrix(hiddenSize, inputSize);
    gWxh->fill(0);

    gWhh = new MfDoubleMatrix(hiddenSize, hiddenSize);
    gWhh->fill(0);

    gWhy = new MfDoubleMatrix(outputSize, hiddenSize);
    gWhy->fill(0);

    gbh = new MfDoubleMatrix(1, hiddenSize);
    gbh->fill(0);

    gby = new MfDoubleMatrix(1, outputSize);
    gby->fill(0);

    //Step 2. Allocate space for times variant variables.
    xAt = new MfDoubleMatrixArray(maxSequenceLength, inputSize, 1);
    hAt = new MfDoubleMatrixArray(maxSequenceLength, 1, hiddenSize);
    yAt = new MfDoubleMatrixArray(maxSequenceLength, outputSize, 1);
    pAt = new MfDoubleMatrixArray(maxSequenceLength, outputSize, 1);
    dxAt = new MfDoubleMatrixArray(maxSequenceLength, inputSize, 1);
    dyAt = new MfDoubleMatrixArray(maxSequenceLength, outputSize, 1);

    //Step 3. Temporary vectors.
    hiddenSizeRowVector = new MfDoubleMatrix(1, hiddenSize);
    hiddenSizeColumnVector = new MfDoubleMatrix(hiddenSize, 1);

    initialized = true;
}//Of initialize

/**
 * Forward.
 * paraData: one-hot coding of a string.
 */
void MfRnnLayer::forward(MfDoubleMatrixArray* paraData)
{
    if(!initialized)
    {
        printf("MfRnnLayer::forward. Layer not initialized yet.\r\n");
        throw "MfRnnLayer::forward. Layer not initialized yet.";
    }//Of if

    for(int i = 1; i < paraData->getLength(); i ++)
    {
        if(paraData->getDataAt(i)->getColumns() != inputSize)
        {
            printf("MfRnnLayer::forward. Data size not match.\r\n");
            throw "MfRnnLayer::forward. Data size not match.";
        }//Of if
    }//Of for i

    //Save the inputs (needed for backPropagation)
    xAt->cloneToMe(paraData);

    //Reset outputs
    lastSequenceLength = xAt->getLength() - 1;

    // hidden state vectors through time
    hAt->setLength(lastSequenceLength + 1);

    // normalized probability vectors through time
    pAt->setLength(lastSequenceLength + 1);

    // normalized probability vectors through time
    yAt->setLength(lastSequenceLength + 1);

    //Copy the current state
    //hAt[0] = new Matrix(h);
    hAt->getDataAt(0)->cloneToMe(h);

    /* Forward pass */
    MfDoubleMatrix* tempHAt;
    MfDoubleMatrix* tempYAt;
    for (int t = 1; t < lastSequenceLength + 1; t ++)
    {
        // find the new hidden state
        //hAt[t] = (Matrix.dot(Wxh, xAt[t])
        //              .add(Matrix.dot(Whh, hAt[t - 1]))
        //              .add(bh))
        //             .tanh();

        //Transpose before multiply.
        hiddenSizeRowVector->cloneToMe(hAt->getDataAt(t - 1));
        hiddenSizeColumnVector->transposeToMe(hiddenSizeRowVector);
        hiddenSizeRowVector->timesToMe(Whh, hiddenSizeColumnVector);
        //Give it another name.
        tempHAt = hAt->getDataAt(t);
        tempHAt->timesToMe(Wxh, xAt->getDataAt(t));
        tempHAt->addToMe(tempHAt, hiddenSizeRowVector);
        tempHAt->addToMe(tempHAt, bh);

        activator->setActivationFunction('t');
        tempHAt->setActivator(activator);
        tempHAt->activate();

        //Find unnormalized output probabilities.
        //yAt[t] = Matrix.dot(Why, hAt[t]).add(by);
        tempYAt = yAt->getDataAt(t);
        tempYAt->timesToMe(Why, tempHAt);
        tempYAt->addToMe(tempYAt, by);

        //Normalize output probabilities
        //pAt[t] = Math.softmax(yAt[t]);
        pAt->getDataAt(t)->softmaxToMe(tempYAt);
    }//Of for t

    /* Update the hidden state */
    h->cloneToMe(hAt->getDataAt(lastSequenceLength));
}//Of forward

/**
 * Get the loss.
 */
MfDoubleMatrixArray* MfRnnLayer::getdy(MfIntArray* paraIntArray)
{
    MfDoubleMatrix* tempMatrix;
    for(int t = 1; t < lastSequenceLength + 1; t ++)
    {
        // backprop into y,
        // http://cs231n.github.io/neural-networks-case-study/#grad

        //dyAt[t].setAt(expected, (dyAt[t].at(expected) - 1));
        int tempExpected = paraIntArray->getValue(t - 1);
        tempMatrix = dyAt->getDataAt(t);

        tempMatrix->setValue(0, tempExpected, tempMatrix->getValue(0, tempExpected) - 1);
    }//Of for t

    return dyAt;
}//Of getdy

/**
 * Get the loss.
 */
double MfRnnLayer::getLoss(MfIntArray* paraIntArray)
{
    double resultLoss = 0;
    //Start at t = 1.
    for (int t = 1; t < lastSequenceLength + 1; t ++)
    {
        //Calculate the cross-entropy loss.
        //loss += -java.lang.Math.log(pAt[t].at(iy[t - 1]));
        resultLoss += -log(pAt->getDataAt(t)->getValue(0, paraIntArray->getValue(t - 1)));
    }//Of for t
    return resultLoss;
}//Of getLoss

/**
 * Unit test.
 */
void MfRnnLayer::unitTest()
{

}//Of unitTest
