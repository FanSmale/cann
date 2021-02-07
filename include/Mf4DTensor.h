/*
 * The C++ Artificial Neural network project.
 * 4D tensor for CNN.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MF4DTENSOR_H
#define MF4DTENSOR_H

#include <Malloc.h>

#include "MfDoubleMatrix.h"

class Mf4DTensor
{
    public:
        //The constructor.
        Mf4DTensor();

        //The constructor.
        Mf4DTensor(int, int, int, int);

        //The destructor.
        virtual ~Mf4DTensor();

        //Fill with one value.
        void fill(double);

        //Convert to string for display.
        string toString();

        //Unit test.
        void unitTest();

    protected:

        //The first dimension length.
        int firstLength;

        //The second dimension length.
        int secondLength;

        //The third dimension length.
        int thirdLength;

        //The fourth dimension length.
        int fourthLength;

        //The data.
        double**** data;

    private:
};

#endif // MF4DTENSOR_H