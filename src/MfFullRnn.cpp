/*
 * The C++ Artificial Neural network project.
 * This class handles the whole RNN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfFullRnn.h"

/**
 * Empty constructor. Not used now.
 */
MfFullRnn::MfFullRnn()
{
    //ctor
}//Of the empty constructor

/**
 * The second constructor.
 * paraSizes: the sizes for layers.
 * paraRate: the learning rate.
 */
MfFullRnn::MfFullRnn(MfIntArray* paraInputSizes, MfIntArray* paraHiddenSizes,
                     MfIntArray* paraOutputSizes, int paraMaxLength, double paraRate)
{
    learningRate = paraRate;

    numLayers = paraInputSizes->getLength();
    layers = new MfRnnLayer* [numLayers];
    for (int i = 0; i < numLayers; i ++)
    {
        layers[i] = new MfRnnLayer(paraInputSizes->getValue(i), paraHiddenSizes->getValue(i),
                                   paraOutputSizes->getValue(i), paraMaxLength, learningRate);
    }//Of for i

}//Of the second constructor


MfFullRnn::~MfFullRnn()
{
    for(int i = 0; i < numLayers; i ++)
    {
        free(layers[i]);
    }
    free(layers);
}//Of the destructor

/**
 * Convert to string for display.
 */
string MfFullRnn::toString()
{
    string resultString = "I am a full RNN with " + to_string(numLayers)
        + " Layers.\r\n";

    return resultString;
}//Of toString
