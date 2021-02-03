/*
 * The C++ Artificial Neural network project.
 * A convolution neural network layer.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

//#include <iostream>
#include "CnnLayer.h"

/**
 * The empty constructor.
 */
CnnLayer::CnnLayer()
{
    //ctor
}//Of the empty constructor.

/**
 * The constructor.
 *
 */
CnnLayer::CnnLayer(int paraNumRows, int paraNumColumns, int paraKernelSize)
{
    //Accept parameters.
    numRows = paraNumRows;
    numColumns = paraNumColumns;
    kernelSize = paraKernelSize;

    //Initialize the kernel.
    kernel.resize(kernelSize, kernelSize);
    for(int i = 0; i < kernelSize; i ++)
    {
        for (int j = 0; j < kernelSize; j ++)
        {
            kernel(i, j) = 1; //random();
        }//Of for j
    }//Of for i
}//Of the second constructor.

/**
 * The destructor.
 */
CnnLayer::~CnnLayer()
{
    //dtor
}//Of the destructor

/**
 * Convolution. Now only the simplest convolution.
 *   For example, a 4*5 image convolution with a 3*3 kernel gets 2*3.
 *   4-3+1=2, and 5-3+1=3.
 * paraMatrix: the input data.
 * Returns: the matrix after convolution.
 */
DoubleMatrix* CnnLayer::convolutionLoseEdge(DoubleMatrix* paraMatrix)
{
    int tempInputNumRows = paraMatrix -> rows();
    int tempInputNumCols = paraMatrix -> cols();

    int tempOutputNumRows = tempInputNumRows - kernelSize + 1;
    int tempOutputNumCols = tempInputNumCols - kernelSize + 1;

    DoubleMatrix* resultMatrixPtr = new DoubleMatrix(tempOutputNumRows, tempOutputNumCols);
    double tempValue;
    for(int i = 0; i < tempOutputNumRows; i ++)
    {
        for(int j = 0; j < tempOutputNumCols; j ++)
        {
            tempValue = 0;
            for(int k = 0; k < kernelSize; k ++)
            {
                for(int k2 = 0; k2 < kernelSize; k2 ++)
                {
                    tempValue += paraMatrix[0](i + k, j + k2) * kernel(k, k2);
                }//Of for k2
            }//Of for k
            resultMatrixPtr[0](i, j) = tempValue;
        }//Of for j
    }//Of for i

    //Free the data from last layer.
    free(paraMatrix);

    return resultMatrixPtr;
}//Of convolutionLoseEdge

/**
 * Convolution. Keep the size of the image.
 *   Inapplicable multiplex will be treated as 0.
 * paraMatrix: the input data.
 * Returns: the matrix after convolution.
 */
DoubleMatrix* CnnLayer::convolutionKeepEdge(DoubleMatrix* paraMatrix)
{
    int tempNumRows = paraMatrix -> rows();
    int tempNumCols = paraMatrix -> cols();

    DoubleMatrix* resultMatrixPtr = new DoubleMatrix(tempNumRows, tempNumCols);
    double tempValue;
    int tempRowIndex, tempColIndex;
    for(int i = 0; i < tempNumRows; i ++)
    {
        for(int j = 0; j < tempNumCols; j ++)
        {
            tempValue = 0;
            for(int k = 0; k < kernelSize; k ++)
            {
                for(int k2 = 0; k2 < kernelSize; k2 ++)
                {
                    tempRowIndex = i + k - kernelSize/2;
                    tempColIndex = j + k2 - kernelSize/2;
                    if ((tempRowIndex >= 0) && (tempRowIndex < tempNumRows)
                        && (tempColIndex >= 0) && (tempColIndex < tempNumCols))
                    {
                        tempValue += paraMatrix[0](tempRowIndex, tempColIndex) * kernel(k, k2);
                    }//Of if
                }//Of for k2
            }//Of for k
            resultMatrixPtr[0](i, j) = tempValue;
        }//Of for j
    }//Of for i

    //Free the data from last layer.
    free(paraMatrix);

    return resultMatrixPtr;
}//Of convolutionKeepEdge

/**
 * Code unit test.
 */
void CnnLayer::unitTest()
{
    CnnLayer tempCnnLayer(4, 5, 3);
    DoubleMatrix* tempMatrixPtr = new DoubleMatrix(4, 5);
    for(int i = 0; i < 4; i ++)
    {
        for(int j = 0; j < 5; j ++)
        {
            tempMatrixPtr[0](i, j) = 1;
        }//Of for j
    }//Of for i

    DoubleMatrix* tempResultPtr = tempCnnLayer.convolutionKeepEdge(tempMatrixPtr);
    printf("Result:\r\n");
    cout << tempResultPtr[0] << endl;
}//Of unitTest

