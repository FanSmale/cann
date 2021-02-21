/*
 * The C++ Artificial Neural network project.
 * This class handles the whole RNN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFFULLRNN_H
#define MFFULLRNN_H

#include "MfIntArray.h"
#include "MfDoubleMatrix.h"
#include "MfRnnLayer.h"

class MfFullRnn
{
    public:
        //The empty constructor. Not used now.
        MfFullRnn();

        //The second constructor.
        MfFullRnn(MfIntArray* paraInputSizes, MfIntArray* paraHiddenSizes,
                  MfIntArray* paraOutputSizes, int paraMaxLength, double paraRate);

        //The destructor.
        virtual ~MfFullRnn();

        //Convert to string for display
        string toString();

        //Setter
        void setRate(double paraRate)
        {
            learningRate = paraRate;
        }

    protected:

        //The number of layers.
        int numLayers;

        //The layers. An array of pointers instead of a 2D matrix.
        MfRnnLayer** layers;

        //The learning rate.
        double learningRate;

    private:
};

#endif // MFFULLRNN_H
