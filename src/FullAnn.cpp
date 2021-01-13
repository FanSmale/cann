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

#include "FullAnn.h"
#include "MfMath.h"

//The default constructor
FullAnn::FullAnn()
{
    numLayers = 0;
    layers = nullptr;
}//Of the default constructor

//The second constructor
FullAnn::FullAnn(IntArray* paraSizes, char paraActivation)
{
    //Allocate space
    numLayers = paraSizes -> getLength() - 1;
    layers = new AnnLayer *[numLayers];
    for (int i = 0; i < numLayers; i ++)
    {
        layers[i] = new AnnLayer(paraSizes -> getValue(i), paraSizes -> getValue(i + 1), paraActivation);
    }//Of for i
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
Matrix* FullAnn::forward(Matrix* paraInput)
{
    return nullptr;
}

//Back propagation
void FullAnn::backpropagation()
{

}

//Code self test
void FullAnn::selfTest()
{
    int tempArray[3] = {3, 5, 7};
    IntArray* tempIntArray = new IntArray(3, tempArray);
    FullAnn* tempFullAnn = new FullAnn(tempIntArray, 's');

    Matrix* tempData = new Matrix(1, 3);
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

}//Of selfTest
