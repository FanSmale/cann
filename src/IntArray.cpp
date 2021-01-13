/*
 * The C++ Artificial Neural network project.
 * Integer array.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "IntArray.h"
#include <Malloc.h>
#include <stdio.h>
#include <string>

//Default constructor
IntArray::IntArray()
{
    length = 0;
    data = nullptr;
}//Of the default constructor

IntArray::IntArray(int paraLength)
{
    length = paraLength;
    data = new int(paraLength);
    for (int i = 0; i < length; i ++)
    {
        data[i] = 0;
    }//Of for i
}//Of the second constructor

IntArray::IntArray(int paraLength, int* paraValues)
{
    length = paraLength;
    data = new int(paraLength);
    for (int i = 0; i < length; i ++)
    {
        data[i] = paraValues[i];
    }//Of for i
}//Of the third constructor

//The destructor
IntArray::~IntArray()
{
    free(data);
}//Of the destructor

//Convert to string
string IntArray::toString()
{
    string resultString = "I am an int array with length " + to_string(length) + "\r\n";
    for (int i = 0; i < length; i ++)
    {
        resultString += to_string(data[i]) + ", ";
    }//Of for i
    resultString += "\r\n data ends. \r\n";

    return resultString;
}//Of toString

//Set one value
void IntArray::setValue(int paraPosition, int paraValue)
{
    if (paraPosition >= length)
    {
        printf("Error occurred in IntArray.setValue(int, int), the index %d is out of bound %d.\r\n",
               paraPosition, length);
        throw -1;
    }
    data[paraPosition] = paraValue;
}//Of setValue

//Get the value at the given position
int IntArray::getValue(int paraPosition)
{
    if (paraPosition >= length)
    {
        printf("Error occurred in IntArray.getValue(int), the index %d is out of bound %d.\r\n",
               paraPosition, length);
        throw -1;
    }
    return data[paraPosition];
}//Of setValue

//Get length
int IntArray::getLength()
{
    return length;
}//Of setValue

//Get a copy
IntArray* IntArray::copy()
{
    IntArray* resultArray = new IntArray(length);

    //Copy
    for (int i = 0; i < length; i ++)
    {
        resultArray -> data[i] = data[i];
    }//Of for i

    return resultArray;
}//Of copy

//Get a copy
void IntArray::copyFrom(IntArray* paraArray)
{
    if (length != paraArray -> length)
    {
        printf("Error occurred in IntArray.copyFrom(IntArray*), lengths do not match: %d vs. %d.\r\n",
               length, paraArray -> length);
    }

    //Copy
    for (int i = 0; i < length; i ++)
    {
        data[i] = paraArray -> data[i];
    }//Of for i
}//Of copyFrom


//Code self test
void IntArray::selfTest()
{
    int tempIntArray[3] = {3, 5, 7};
    IntArray* tempArray = new IntArray(3, tempIntArray);

    printf("Original data\r\n");
    printf(tempArray -> toString().data());

    tempArray -> setValue(1, 9);
    printf("Set one value\r\n");
    printf(tempArray -> toString().data());
}//Of selfTest



