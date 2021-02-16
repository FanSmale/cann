/**
 * The C++ Artificial Neural network project.
 * Stack a number of MfCnnLayer to form a network.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFFULLCNN_H
#define MFFULLCNN_H

#define MAX_NUM_CNN_LAYERS 100

#include "MfCnnLayer.h"
#include "MfDataReader.h"

class MfFullCnn
{
    public:
        //The default constructor.
        MfFullCnn();

        //The second constructor.
        MfFullCnn(int paraBatchSize);

        //The destructor.
        virtual ~MfFullCnn();

        //Add a layer.
        void addLayer(int paraLayerType, int paraNum, MfSize* paraSize);

        //Setup for train.
        void setup();

        //Initialize the input layer output.
        void setInputLayerOutput(MfDoubleMatrix* paraData);

        //Forward.
        int forward(MfDoubleMatrix* paraData);

        //Back propagation.
        void backPropagation(int paraLabel);

        //Update parameters.
        void updateParameters();

        //Initialize the random array.
        void initializeRandomArray(int paraLength);

        //Re-randomize.
        void randomize();

        //Prepare for a new batch.
        void prepareForNewBatch();

        //Prepare for a new record.
        void prepareForNewRecord();

        //Output kernels for debugging.
        void outputKernelsToFile();

        //Train.
        double train(MfDoubleMatrix* paraX, MfIntArray* paraY);

        //Test.
        double test(MfDoubleMatrix* paraX, MfIntArray* paraY);

        //Training/testing.
        void mnistTest();

        //Unit test, essentially integrated test.
        void integratedTest();

    protected:

        //The number of layers.
        int numLayers;

        //The layers.
        MfCnnLayer** layers;

        //The batch size
        int batchSize;

        //The activator
        Activator* layerActivator;

        //The random array for training.
        MfIntArray* randomArray;

    private:
};

#endif // MFFULLCNN_H
