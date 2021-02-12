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

#define ALPHA 0.05
#define LAMBDA 0.00

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
        MfCnnLayer(int, int, int, MfSize*);

        //The destructor
        virtual ~MfCnnLayer();

        //Initialize the kernel.
        void initKernel(int);

        //Initialize the output kernel. The essential difference from initKernel is unknown.
        void initOutKernel(int, MfSize*);

        //Initialize the bias.
        void initBias();

        //Initialize the errors.
        void initErrors();

        //Initialize the output maps.
        void initOutMaps();

        //Prepare for new batch.
        void prepareForNewBatch();

        //Prepare for new record.
        void prepareForNewRecord();

        //Getter.
        int getNumClasses();

        //Getter.
        int getLayerType();

        //Getter.
        int getNumOutMaps();

        //Setter.
        void setNumOutMaps(int);

        //Getter.
        MfSize* getMapSize();

        //Setter.
        void setMapSize(MfSize* paraSize);

        //Getter.
        MfSize* getKernelSize();

        //Setter.
        void setKernelSize(MfSize* paraSize);

        //Getter.
        MfSize* getScaleSize();

        //Setter.
        void setScaleSize(MfSize* paraSize);

        //Getter.
        MfDoubleMatrix* getKernelAt(int paraFrontMap, int paraOutMap);

        //Setter.
        void setKernelAt(int paraFrontMap, int paraOutMap, MfDoubleMatrix* paraKernel);

        //Getter.
        MfDoubleMatrix* getRot180KernelAt(int paraFrontMap, int paraOutMap);

        //Getter.
        double getBiasAt(int paraMapNo);

        //Setter.
        void setBiasAt(int paraMapNo, double paraValue);

        //Getter.
        Mf4DTensor* getOutMaps();

        //Set the out map value.
        void setOutMapValue(int paraMapNo, int paraX, int paraY, double paraValue);

        //Set the map value.
        void setOutMapValue(int paraMapNo, MfDoubleMatrix* paraMatrix);

        //Getter.
        MfDoubleMatrix* getOutMapAt(int paraIndex);

        //Getter.
        MfDoubleMatrix* getOutMapAt(int paraRecordId, int paraOutMapNo);

        //Getter.
        Mf4DTensor* getErrors();

        //Getter.
        MfDoubleMatrix* getErrorsAt(int paraMapNo);

        //Setter.
        void setErrorsAt(int, MfDoubleMatrix*);

        //Getter.
        MfDoubleMatrix* getErrorsAt(int paraRecordId, int paraMapNo);

        //Setter.
        void setErrorAt(int, int, int, double);

        //Setup.
        void setup();

        //Setter.
        void setLastLayer(MfCnnLayer* paraLayer);

        //Setter.
        void setNextLayer(MfCnnLayer* paraLayer);

        //Handle the input layer.
        void setInputLayerOutput(MfDoubleMatrix* paraData);

        //Handle the convolution layer.
        void setConvolutionOutput();

        //Handle the sampling layer.
        void setSamplingOutput();

        //Get the prediction for the current instance.
        int getCurrentPrediction();

        //Forward an instance, the parameters may not be useful.
        void forward(MfDoubleMatrix* paraData);

        //Set the errors of the convolution layer.
        void setConvolutionLayerErrors();

        //Set the errors of the sampling layer.
        void setSamplingLayerErrors();

        //Set the error of the output layer.
        void setOutputLayerErrors(int paraLabel);

        //Back propagation, the parameters may not be useful.
        void backPropagation(int paraLabel);

        //Update kernels.
        void updateKernels();

        //Update bias.
        void updateBias();

        //Set the layer activator.
        void setLayerActivator(char);

        //Unit test.
        void unitTest();

        //Getter.
        MfDoubleMatrix* getCurrentOutMap();

    protected:

        //The number of classes, not the index of the class (label) attribute.
        int numClasses = -1;

        //The batch size.
        int batchSize;

        //For batch processing.
        int recordInBatch = 0;

        //Layer type, 4 types.
        int layerType;

        //The number of output maps.
        int numOutMaps;

        //The size of output maps.
        MfSize* mapSize;

        //The kernel size.
        MfSize* kernelSize;

        //The scale size.
        MfSize* scaleSize;

        //Kernel. Dimensions: [front map][out map][width][height].
        Mf4DTensor* kernel;

        //The current kernel.
        MfDoubleMatrix* currentKernel;

        //Delta kernel.
        MfDoubleMatrix* deltaKernel;

        //Single delta kernel.
        MfDoubleMatrix* singleDeltaKernel;

        //The current rotate 180 kernel. Avoid local variables.
        MfDoubleMatrix* currentRot180Kernel;

        //Bias. The length is outMapNum.
        MfDoubleMatrix* bias;

        //Out maps. Dimensions: [batchSize][numOutMaps][mapSize.width][mapSize.height].
        //The first dimension is due to the parallel computing and parameter updating.
        Mf4DTensor* outMaps;

        //Errors. Dimensions: [batchSize][numOutMaps][mapSize.width][mapSize.height].
        Mf4DTensor* errors;

        //Current errors, for one map, one instance in a batch.
        MfDoubleMatrix* currentErrors;

        //Current single errors, for accumulation.
        MfDoubleMatrix* currentSingleErrors;

        //layers[i - 1].
        MfCnnLayer* lastLayer;

        //layers[i + 1].
        MfCnnLayer* nextLayer;

    private:

        //The activator of this layer.
        Activator* layerActivator;

        //Current out map.
        MfDoubleMatrix* currentOutMap;

        //A single output, space allocated once and used many times to avoid allocating for temporary variables.
        MfDoubleMatrix* singleOutMap;

};

#endif // MFCNNLAYER_H
