/*
 * The C++ Artificial Neural network project.
 * This class manages an array of MfDoubleMatrix.
 * It is used in MfRnnLayer.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfDoubleMatrixArray.h"

/**
 * The empty constructor. Not used now.
 */
MfDoubleMatrixArray::MfDoubleMatrixArray()
{
    maxLength = 0;
    //ctor
}//Of the empty constructor

/**
 * The second constructor.
 * paraMaxLength: the maximal length of the array.
 * paraRows: the rows of each matrix.
 * paraColumns: the columns of each matrix.
 */
MfDoubleMatrixArray::MfDoubleMatrixArray(int paraMaxLength, int paraRows, int paraColumns)
{
    maxLength = paraMaxLength;

    data = new MfDoubleMatrix*[maxLength];
    for(int i = 0; i < maxLength; i ++)
    {
        data[i] = new MfDoubleMatrix(paraRows, paraColumns);
    }//Of for i

    length = 0;
}//Of the second constructor

/**
 * The destructor.
 */
MfDoubleMatrixArray::~MfDoubleMatrixArray()
{
    //dtor
}//Of the destructor

/**
 * Convert to string for display.
 * Returns: The string showing myself.
 */
string MfDoubleMatrixArray::toString()
{
    string resultString = "MfDoubleMatrixArray with maximal length "
        + to_string(maxLength) + " and valid length " + to_string(length) + "\r\n";
    for(int i = 0; i < maxLength; i ++)
    {
        resultString += to_string(i) + ":\r\n" + data[i]->toString();
    }//Of for i

    //resultString += "matrix ends. \r\n";

    return resultString;
}//Of toString

/**
 * One hot coding.
 */
MfDoubleMatrixArray* MfDoubleMatrixArray::cloneToMe(MfDoubleMatrixArray* paraArray)
{
    if(paraArray->length > maxLength)
    {
        printf("Error in MfDoubleMatrixArray::cloneToMe, the given array is too long.\r\n");
        throw "Error in MfDoubleMatrixArray::cloneToMe, the given array is too long.\r\n";
    }//Of if

    if((paraArray->getDataAt(0)->getRows() != data[0]->getRows()) ||
       (paraArray->getDataAt(0)->getColumns() != data[0]->getColumns()))
    {
        printf("Error in MfDoubleMatrixArray::cloneToMe, MfDoubleMatrix sizes do not match.\r\n");
        throw "Error in MfDoubleMatrixArray::cloneToMe, MfDoubleMatrix sizes do not match.\r\n";
    }//Of if

    //The actual length
    length = paraArray->length;
    for(int i = 0; i < paraArray->length; i ++)
    {
        for(int j = 0; j < paraArray->getDataAt(i)->getRows(); j ++)
        {
            for(int k = 0; k < paraArray->getDataAt(i)->getColumns(); k++)
            {
                data[i]->setValue(j, k, paraArray->getDataAt(i)->getValue(j, k));
            }//Of for k
        }//Of for j
    }//Of for i

    return this;
}//Of cloneToMe

/**
 * One hot coding.
 */
MfDoubleMatrixArray* MfDoubleMatrixArray::oneHotToMe(MfIntArray* paraIndices)
{
    if(paraIndices->getLength() > length - 1)
    {
        printf("Error in MfDoubleMatrixArray::oneHotToMe, the given array is too long.\r\n");
        throw "Error in MfDoubleMatrixArray::oneHotToMe, the given array is too long.\r\n";
    }//Of if

    //Starting from 1.
    for(int i = 0; i < paraIndices->getLength(); i ++)
    {
        data[i + 1]->oneHotToMe(paraIndices->getValue(i));
    }//Of for i

    return this;
}//Of oneHotToMe

/**
 * Unit test.
 */
void MfDoubleMatrixArray::unitTest()
{
    printf("MfDoubleMatrixArray unit test\r\n");
    MfDoubleMatrixArray* tempArray1 = new MfDoubleMatrixArray(3, 2, 2);
    tempArray1->setLength(3);
    printf(tempArray1->toString().data());

    printf("Matrix #1\r\n");
    printf(tempArray1->getDataAt(1)->toString().data());

    MfDoubleMatrixArray* tempArray2 = new MfDoubleMatrixArray(5, 1, 9);
    tempArray2->setLength(5);
    MfIntArray* tempIntArray = new MfIntArray(3);
    tempIntArray->setValue(0, 2);
    tempIntArray->setValue(1, 0);
    tempIntArray->setValue(2, 1);
    tempArray2->oneHotToMe(tempIntArray);
    printf("One hot:\r\n");
    printf(tempArray2->toString().data());

    MfDoubleMatrixArray* tempArray3 = new MfDoubleMatrixArray(4, 2, 2);
    tempArray3->cloneToMe(tempArray1);
    printf("After clone\r\n");
    printf(tempArray3->toString().data());

}//Of unitTest
