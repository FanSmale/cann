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

#ifndef CNNLAYER_H
#define CNNLAYER_H

#include <string>
#include <iostream>
#include <stdio.h>
#include "EigenSupport.h"

using namespace std;

class CnnLayer
{
    public:
        //The empty constructor.
        CnnLayer();

        //The second constructor.
        CnnLayer(int paraNumRows, int paraNumColumns, int paraKernelSize);

        //The destructor.
        virtual ~CnnLayer();

        //Convolution. Lose the edge.
        DoubleMatrix* convolutionLoseEdge(DoubleMatrix* paraMatrix);

        //Convolution. Keep edge.
        DoubleMatrix* convolutionKeepEdge(DoubleMatrix* paraMatrix);

        //Code unit test
        void unitTest();

    protected:

        //The number of rows.
        int numRows;

        //The number of columns.
        int numColumns;

        //The size of the kernel.
        int kernelSize;

        //The kernel.
        DoubleMatrix kernel;

    private:
};

#endif // CNNLAYER_H
