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
        MfCnnLayer(int, int, int, MfSize*);

        //The destructor
        virtual ~MfCnnLayer();

        //Initialize the kernel.
        void initKernel(int);

        //Initialize the output kernel. The essential difference from initKernel is unknown.
        //void initOutKernel(int, MfSize*);

        //Initialize the bias.
        void initBias();

        //Initialize the errors.
        void initErrors();

        //Initialize the output maps.
        void initOutMaps();

        //Prepare for new batch.
        void prepareForNewBatch()
        {
            recordInBatch = 0;
        }

        //Prepare for new record.
        void prepareForNewRecord()
        {
            recordInBatch ++;
        }

        //Getter.
        int getNumClasses()
        {
            return numClasses;
        }

        //Getter.
        int getLayerType()
        {
            return layerType;
        }

        //Getter.
        int getNumOutMaps()
        {
            return numOutMaps;
        }

        //Setter.
        void setNumOutMaps(int paraNum)
        {
            numOutMaps = paraNum;
        }

        //Getter.
        MfSize* getMapSize()
        {
            return mapSize;
        }

        //Setter.
        void setMapSize(MfSize* paraSize)
        {
            mapSize->cloneToMe(paraSize);
        }

        //Getter.
        MfSize* getKernelSize()
        {
            return kernelSize;
        }

        //Setter.
        void setKernelSize(MfSize* paraSize)
        {
            kernelSize->cloneToMe(paraSize);
        }

        //Getter.
        MfSize* getScaleSize()
        {
            return scaleSize;
        }

        //Setter.
        void setScaleSize(MfSize* paraSize)
        {
            scaleSize->cloneToMe(paraSize);
        }

        //Getter.
        double getAlpha()
        {
            return alpha;
        }

        //Setter.
        void setAlpha(double paraAlpha)
        {
            alpha = paraAlpha;
        }

        //Getter.
        double getLambda()
        {
            return lambda;
        }

        //Setter.
        void setLambda(double paraLambda)
        {
            lambda = paraLambda;
        }

        //Update alpha, the mechanism is unknown.
        void updateAlpha()
        {
            alpha = alpha * 0.9 + 0.001;
        }

        //Getter.
        MfDoubleMatrix* getKernelAt(int paraFrontMap, int paraOutMap);

        //Getter.
        Mf4DTensor* getKernel()
        {
            return kernel;
        }

        //Setter.
        void setKernelAt(int paraFrontMap, int paraOutMap, MfDoubleMatrix* paraKernel);

        //Getter.
        MfDoubleMatrix* getRot180KernelAt(int paraFrontMap, int paraOutMap);

        //Getter.
        double getBiasAt(int paraMapNo)
        {
            return bias->getValue(0, paraMapNo);
        }

        //Setter.
        void setBiasAt(int paraMapNo, double paraValue)
        {
            bias->setValue(0, paraMapNo, paraValue);
        }

        //Getter.
        Mf4DTensor* getOutMaps()
        {
            return outMaps;
        }

        //Set the out map value.
        void setOutMapValue(int paraMapNo, int paraX, int paraY, double paraValue);

        //Set the map value.
        void setOutMapValue(int paraMapNo, MfDoubleMatrix* paraMatrix);

        //Getter.
        MfDoubleMatrix* getOutMapAt(int paraIndex);

        //Getter.
        MfDoubleMatrix* getOutMapAt(int paraRecordId, int paraOutMapNo);

        //Getter.
        Mf4DTensor* getErrors()
        {
            return errors;
        }

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
        void setLastLayer(MfCnnLayer* paraLayer)
        {
            lastLayer = paraLayer;
        }

        //Setter.
        void setNextLayer(MfCnnLayer* paraLayer)
        {
            nextLayer = paraLayer;
        }

        //Handle the input layer.
        void setInputLayerOutput(MfDoubleMatrix* paraData);

        //Handle the convolution layer.
        void setConvolutionOutput();

        //Handle the sampling layer.
        void setSamplingOutput();

        //Get the prediction for the current instance.
        int getCurrentPrediction();

        //Get the prediction distribution for the current instance.
        MfDoubleMatrix* getCurrentPredictionDistribution()
        {
            return predictionDistribution;
        }

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

        //Getter.
        MfDoubleMatrix* getCurrentOutMap()
        {
            return currentOutMap;
        }

        //Unit test.
        void unitTest();

    protected:

        //Layer type, 4 types.
        int layerType;

        //The number of classes, not the index of the class (label) attribute.
        int numClasses = -1;

        //The batch size.
        int batchSize;

        //For batch processing.
        int recordInBatch = 0;

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

        //Current out map.
        MfDoubleMatrix* currentOutMap;

        //A single output, space allocated once and used many times to avoid allocating for temporary variables.
        MfDoubleMatrix* singleOutMap;

        //The prediction distribution. For debugging.
        MfDoubleMatrix* predictionDistribution;

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

        //For parameter updating, however not learning rate.
        double alpha;

        //Also for parameter updating.
        double lambda;
};

#endif // MFCNNLAYER_H
