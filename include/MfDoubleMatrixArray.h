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

#ifndef MFDOUBLEMATRIXARRAY_H
#define MFDOUBLEMATRIXARRAY_H

#include "MfDoubleMatrix.h"
#include "MfIntArray.h"

class MfDoubleMatrixArray
{
    public:

        //The default constructor. Not used now.
        MfDoubleMatrixArray();

        //The second constructor.
        MfDoubleMatrixArray(int, int, int);

        //The destructor.
        virtual ~MfDoubleMatrixArray();

        //Convert to string for display.
        string toString();

        //Setter.
        void setLength(int paraLength)
        {
            length = paraLength;
        }

        //Getter.
        int getLength()
        {
            return length;
        }

        //Length increase by 1.
        void lengthIncrease()
        {
            length ++;
        }

        MfDoubleMatrix** getData()
        {
            return data;
        }

        MfDoubleMatrix* getDataAt(int paraIndex)
        {
            return data[paraIndex];
        }

        //Clone to me.
        MfDoubleMatrixArray* cloneToMe(MfDoubleMatrixArray* paraArray);

        //One hot coding.
        MfDoubleMatrixArray* oneHotToMe(MfIntArray* paraIndices);

        //Unit test.
        void unitTest();

    protected:

        //The data.
        MfDoubleMatrix** data;

        //The maximal length.
        int maxLength;

        //The actual length.
        int length;

    private:

};

#endif // MFDOUBLEMATRIXARRAY_H
