/*
 * The C++ Artificial Neural network project.
 * Integer array.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, Southwest Petroleum University
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef INTARRAY_H
#define INTARRAY_H

#include <string>

using namespace std;

class IntArray
{
public:
    //Default constructor
    IntArray();

    //Initialize it with the given length
    IntArray(int paraLength);

    //Initialize it with the given array
    IntArray(int paraLength, int* paraValues);

    //Destructor
    virtual ~IntArray();

    //Convert to string
    string toString();

    //Set one value
    void setValue(int paraPosition, int paraValue);

    //Get the value at the given position
    int getValue(int paraPosition);

    //Get length. No setLength enabled
    int getLength();

    //Get a copy
    IntArray* copy();

    //Copy from another array
    void copyFrom(IntArray* paraArray);

    //Code unit test
    void selfTest();

protected:
    //The length of the array
    int length;

    //The data
    int* data;

private:
};

#endif // INTARRAY_H
