/*
 * The C++ Artificial Neural network project.
 * Integer array.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, Southwest Petroleum University
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFINTARRAY_H
#define MFINTARRAY_H

#include <string>
#include <Malloc.h>
#include <stdio.h>
#include <random>
#include <fstream>
#include <iostream>

using namespace std;
using std::default_random_engine;

class MfIntArray
{
public:
    //Default constructor
    MfIntArray();

    //Initialize it with the given length
    MfIntArray(int paraLength);

    //Initialize it with the given array
    MfIntArray(int paraLength, int* paraValues);

    //Destructor
    virtual ~MfIntArray();

    //Convert to string
    string toString();

    //Set one value
    void setValue(int paraPosition, int paraValue);

    //Get the value at the given position
    int getValue(int paraPosition);

    //Get length. No setLength enabled
    int getLength();

    //Get a copy
    MfIntArray* copy();

    //Copy from another array
    void copyFrom(MfIntArray* paraArray);

    //Randomize the order
    void randomizeOrder();

    //Code unit test
    void unitTest();

protected:
    //The length of the array
    int length;

    //The data
    int* data;

private:
};

#endif // MFINTARRAY_H
