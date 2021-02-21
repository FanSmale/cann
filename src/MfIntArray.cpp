/*
 * The C++ Artificial Neural network project.
 * Integer array with my own implementation. Avoid using other packages.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfIntArray.h"

/**
 * Default constructor.
 */
MfIntArray::MfIntArray()
{
    length = 0;
    data = nullptr;
}//Of the default constructor

/**
 * The second constructor. The values are [0, 1, 2, ..., length - 1].
 * paraLength: the length of the array.
 */
MfIntArray::MfIntArray(int paraLength)
{
    length = paraLength;
    data = new int[paraLength];

    for (int i = 0; i < length; i ++)
    {
        data[i] = i;
    }//Of for i
}//Of the second constructor

/**
 * The third constructor.
 * paraLength: the length of the array.
 * paraValues: the given values.
 */
MfIntArray::MfIntArray(int paraLength, int* paraValues)
{
    length = paraLength;
    data = new int(paraLength);
    for (int i = 0; i < length; i ++)
    {
        data[i] = paraValues[i];
    }//Of for i
}//Of the third constructor

/**
 * The destructor.
 */
MfIntArray::~MfIntArray()
{
    free(data);
}//Of the destructor

/**
 * Convert to string for display.
 * Return a string showing myself.
 */
string MfIntArray::toString()
{
    string resultString = "I am an int array with length " + to_string(length) + "\r\n";
    for (int i = 0; i < length; i ++)
    {
        resultString += to_string(data[i]) + ", ";
    }//Of for i
    resultString += "\r\n data ends. \r\n";

    return resultString;
}//Of toString

/**
 * Setter.
 * paraPosition: the position in the array.
 * paraValue: the new value.
 */
void MfIntArray::setValue(int paraPosition, int paraValue)
{
    if (paraPosition >= length)
    {
        printf("Error occurred in MfIntArray.setValue(int, int), the index %d is out of bound %d.\r\n",
               paraPosition, length);
        throw -1;
    }
    data[paraPosition] = paraValue;
}//Of setValue

/**
 * Getter.
 * paraPosition: the position in the array.
 * Return: the value at the position.
 */
int MfIntArray::getValue(int paraPosition)
{
    if (paraPosition >= length)
    {
        printf("Error occurred in MfIntArray.getValue(int), the index %d is out of bound %d.\r\n",
               paraPosition, length);
        throw -1;
    }
    return data[paraPosition];
}//Of setValue

/**
 * Getter.
 * Return: a copy of this array.
MfIntArray* MfIntArray::copy()
{
    MfIntArray* resultArray = new MfIntArray(length);

    //Copy
    for (int i = 0; i < length; i ++)
    {
        resultArray->data[i] = data[i];
    }//Of for i

    return resultArray;
}//Of copy
 */

/**
 * Copy from another array.
 * paraArray: the given array.
 */
void MfIntArray::copyFrom(MfIntArray* paraArray)
{
    if (length != paraArray->length)
    {
        printf("Error occurred in MfIntArray.copyFrom(MfIntArray*), lengths do not match: %d vs. %d.\r\n",
               length, paraArray->length);
    }

    //Copy
    for (int i = 0; i < length; i ++)
    {
        data[i] = paraArray->data[i];
    }//Of for i
}//Of copyFrom


/**
 * Randomize the order.
 */
void MfIntArray::randomizeOrder()
{
    int tempFirstIndex, tempSecondIndex, tempValue;

    for(int i = 0; i < length * 10; i++)
    {
        //tempFirstIndex = randomEngine() % length;
        //tempSecondIndex = randomEngine() % length;
        tempFirstIndex = rand() % length;
        tempSecondIndex = rand() % length;


        tempValue = data[tempFirstIndex];
        data[tempFirstIndex] = data[tempSecondIndex];
        data[tempSecondIndex] = tempValue;
    }//Of for i
}//Of randomizeOrder

/**
 * Code unit test.
 */
void MfIntArray::unitTest()
{
    int tempMfIntArray[3] = {3, 5, 7};
    MfIntArray* tempArray = new MfIntArray(3, tempMfIntArray);

    printf("Original data\r\n");
    printf(tempArray->toString().data());

    tempArray->setValue(1, 9);
    printf("Set one value\r\n");
    printf(tempArray->toString().data());
}//Of unitTest



