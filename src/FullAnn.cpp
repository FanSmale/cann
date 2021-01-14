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
FullAnn::FullAnn(IntArray* paraSizes, char paraActivation)
{
    printf("Test the constructor of FullAnn.cpp\r\n");
    //Allocate space
    numLayers = paraSizes -> getLength() - 1;
    layers = new AnnLayer* [numLayers];
    printf("layers allocated \r\n");
    for (int i = 0; i < numLayers; i ++)
    {
        printf("layer %d \r\n", i);
        layers[i] = new AnnLayer(paraSizes -> getValue(i), paraSizes -> getValue(i + 1), paraActivation);
    }//Of for i
    printf("End of the constructor of FullAnn.cpp\r\n");
}//Of the second constructor

//Destructor
FullAnn::~FullAnn()
{
    //dtor
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
        printf("After layer %d.\r\n", i);
        cout << tempData <<endl;
        //printf(tempData -> toString().data());
    }//Of for i
    return tempData;
}//Of forward

//Back propagation
void FullAnn::backpropagation()
{

}

//Code self test
void FullAnn::selfTest()
{
    printf("Test FullAnn.cpp\r\n");
    int tempArray[3] = {3, 5, 7};
    IntArray* tempIntArray = new IntArray(3, tempArray);

    printf("IntArray constructed. \r\n");
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's');

    printf("FullAnn built\r\n");

    DoubleMatrix tempData;
    tempData.resize(1, 3);
    tempData << 1.2, 1.6, 2.7;
    printf("Input data built\r\n");

    tempData = tempFullAnn -> forward(tempData);
    printf("After forward \r\n");

    cout << tempData <<endl;
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
