/*
 * The C++ Artificial Neural network project.
 * Manages all four types of CNN layers.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFCNNLAYER_H
#define MFCNNLAYER_H

#define INPUT_LAYER 0
#define CONVOLUTION_LAYER 1
#define SAMPLING_LAYER 2
#define OUTPUT_LAYER 3

#include <Malloc.h>

#include "MfDoubleMatrix.h"
#include "MfSize.h"
#include "Mf4DTensor.h"

class MfCnnLayer
{
    public:
        //The default constructor.
        MfCnnLayer();

        //The second constructor.
        MfCnnLayer(int, int, MfSize*);

        //The destructor
        virtual ~MfCnnLayer();

        //Initialize the kernel.
        void initKernel(int);

        //Initialize the output kernel. The essential difference from initKernel is unknown.
        void initOutputKernel(int, MfSize*);

        //Initialize the bias.
        void initBias();

        //Initialize the errors.
        void initErrors(int);

    protected:

        //Layer type, 4 types.
        int layerType;

        //The number of output maps.
        int numOutputMaps;

        //The map size.
        MfSize* mapSize;

        //The kernel size.
        MfSize* kernelSize;

        //The scale size.
        MfSize* scaleSize;

        //The number of classes, not the index of the class (label) attribute.
        int numClasses = -1;

        //Kernel. Dimensions: [front map][out map][width][height].
        Mf4DTensor* kernel;

        //Bias. The length is outMapNum.
        MfDoubleMatrix* bias;

        //Out maps. Dimensions: [batch size][outMapNum][mapSize.width][mapSize.height].
        Mf4DTensor* outMaps;

        //Errors. I don't know the dimension yet.
        Mf4DTensor* errors;

        //For batch processing.
        int recordInBatch = 0;

    private:
};

#endif // MFCNNLAYER_H
