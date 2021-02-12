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

#include "MfFullCnn.h"

/**
 * The default constructor.
 */
MfFullCnn::MfFullCnn()
{
    batchSize = 1;
    numLayers = 0;

    //Just allocate some pointers.
    layers = new MfCnnLayer*[MAX_NUM_CNN_LAYERS];

    layerActivator = new Activator('s');
}//Of the default constructor

/**
 * The second constructor.
 * paraLayerTypes: layer types in integer string, e.g., "012123" where 0, 1, 2 and 3
 *   stand INPUT, CONVOLUTION, MAPPING and OUTPUT, respectively.
 * paraInputSize: the size of the input.
 * paraBatchSize: the batch size.
 */
MfFullCnn::MfFullCnn(int paraBatchSize)
{
    batchSize = paraBatchSize;
    numLayers = 0;

    //Just allocate some pointers.
    layers = new MfCnnLayer*[MAX_NUM_CNN_LAYERS];

    layerActivator = new Activator('s');
}//Of the second constructor

/**
 * Add a layer. Layers are added one by one because it is complex.
 * paraLayerTypes: layer types in integer.
 * paraNum: differs for different type.
 * paraSize: differs for different type.
 */
void MfFullCnn::addLayer(int paraLayerType, int paraNum, MfSize* paraSize)
{
    MfCnnLayer* tempLayer = new MfCnnLayer(paraLayerType, batchSize, paraNum, paraSize);
    //Default activator is sigmoid.
    tempLayer->setLayerActivator('s');
    layers[numLayers] = tempLayer;

    if (numLayers > 0)
    {
        tempLayer->setLastLayer(layers[numLayers - 1]);
        layers[numLayers - 1]->setNextLayer(tempLayer);
    }//Of if
    numLayers ++;
}//Of addLayer

/**
 * The destructor.
 */
MfFullCnn::~MfFullCnn()
{
    //dtor
}//Of the destructor

/**
 * Setup.
 */
void MfFullCnn::setup()
{
    for(int i = 0; i < numLayers; i ++)
    {
        printf("Layer[%d] setup\r\n", i);
        layers[i]->setup();
    }//Of for i
}//Of setup

/**
 * Forward computing. Similar to ANN, invoke forward of each layer.
 * paraData: the given training instance.
 * paraLabel: the label of the instance.
 */
int MfFullCnn::forward(MfDoubleMatrix* paraData) {
    for (int i = 0; i < numLayers; i++)
    {
        //printf("Layer[%d] forward\r\n", i);
        layers[i]->forward(paraData);
    }//Of for i

    int tempPrediction = layers[numLayers - 1]->getCurrentPrediction();
    printf("%d ", tempPrediction);
    return tempPrediction;
}//Of forward

/**
 * Back propagation. Similar to ANN, invoke forward of each layer.
 * paraData: the given training instance.
 * paraLabel: the label of the instance.
 */
void MfFullCnn::backPropagation(int paraLabel)
{
    //Layer 0 is not considered.
    for (int i = numLayers - 1; i > 0; i--)
    {
        //printf("Layer[%d] backPropagation\r\n", i);
        layers[i]->backPropagation(paraLabel);
    }//Of for i
}//Of backPropagation

/**
 * Update parameters.
 */
void MfFullCnn::updateParameters()
{
    for (int i = 1; i < numLayers; i++)
    {
        switch (layers[i]->getLayerType())
        {
        case CONVOLUTION_LAYER:
        case OUTPUT_LAYER:
            layers[i]->updateKernels();
            layers[i]->updateBias();
            break;
        default:
            break;
        }//Of switch
    }//Of for i
}//Of updateParameters

/**
 * Initialize the random array.
 */
void MfFullCnn::initializeRandomArray(int paraLength)
{
    randomArray = new MfIntArray(paraLength);
    randomArray->randomizeOrder();
}//Of initializeRandomArray

/**
 * Randomize.
 */
void MfFullCnn::randomize()
{
    randomArray->randomizeOrder();
}//Of randomize

/**
 * Prepare for a new batch.
 */
void MfFullCnn::prepareForNewBatch()
{
    for(int i = 0; i < numLayers; i ++)
    {
        layers[i] -> prepareForNewBatch();
    }//Of for i
}//Of prepareForNewBatch

/**
 * Prepare for a new batch.
 */
void MfFullCnn::prepareForNewRecord()
{
    for(int i = 0; i < numLayers; i ++)
    {
        layers[i] -> prepareForNewRecord();
    }//Of for i
}//Of prepareForNewBatch

/**
 * Train the network.
 * paraData: the given training instances.
 * paraLabel: the labels of the instances.
 */
void MfFullCnn::train(MfDoubleMatrix* paraX, MfIntArray* paraY)
{
    int tempRows = paraX->getRows();
    int tempColumns = paraX->getColumns();
    int tempEpocs = tempRows / batchSize;

    int tempInstance;
    int tempLabel;
    MfDoubleMatrix* tempData = new MfDoubleMatrix(1, tempColumns);
    //printf("Start to train CNN\r\n");

    randomize();
    for(int e = 0; e < tempEpocs; e ++)
    {
        //printf("e = %d\r\n", e);
        //A new batch
        prepareForNewBatch();
        for(int i = 0; i < batchSize; i ++)
        {
            //printf("i = %d\r\n", i);
            tempInstance = randomArray->getValue(e * batchSize + i);
            for(int j = 0; j < tempColumns; j ++)
            {
                tempData->setValue(0, j, paraX->getValue(tempInstance, j));
            }//Of for j
            tempLabel = paraY->getValue(tempInstance);
            forward(tempData);
            backPropagation(tempLabel);

            //A new record
            prepareForNewRecord();
        }//Of for i

        //Update for each batch
        //printf("\r\n updateParameters\r\n");
        updateParameters();
    }//Of for e
}//Of train

/**
 * Test the network.
 * paraData: the given testing instances.
 * paraLabel: the labels of the instances.
 * Return: the prediction accuracy.
 */
double MfFullCnn::test(MfDoubleMatrix* paraX, MfIntArray* paraY)
{
    int tempRows = paraX->getRows();
    int tempColumns = paraX->getColumns();
    int tempLabel;
    int tempPrediction;
    double tempNumCorrect = 0;
    MfDoubleMatrix* tempData = new MfDoubleMatrix(1, tempColumns);
    printf("Start to test CNN\r\n");

    prepareForNewBatch();
    for(int i = 0; i < 10; i ++)
    {
        //printf("i = %d\r\n", i);
        for(int j = 0; j < tempColumns; j ++)
        {
            tempData->setValue(0, j, paraX->getValue(i, j));
        }//Of for j
        tempLabel = paraY->getValue(i);

        tempPrediction = forward(tempData);
        if (tempPrediction == tempLabel)
        {
            tempNumCorrect++;
            printf("Correct\r\n");
        }  else
        {
            printf("Incorrect: %d vs. %d.\r\n", tempPrediction, tempLabel);
        }//Of if
    }//Of for i

    return tempNumCorrect/tempRows;
}//Of test

/**
 * Training/testing test.
 * The digit recognition data.
 */
void MfFullCnn::trainingTestingTest()
{
    //Step 1. Build the CNN
    printf("Building CNN\r\n");
    MfFullCnn* tempCnn = new MfFullCnn(10);
    MfSize* tempImageSize = new MfSize(28, 28);
    MfSize* tempConvolutionSize = new MfSize(5, 5);
    MfSize* tempSampleSize = new MfSize(2, 2);
    int tempNumClasses = 10;

    //Four layers
    printf("Adding layers\r\n");
    tempCnn->addLayer(INPUT_LAYER, -1, tempImageSize);
    tempCnn->addLayer(CONVOLUTION_LAYER, 6, tempConvolutionSize);
    tempCnn->addLayer(SAMPLING_LAYER, -1, tempSampleSize);
    tempCnn->addLayer(CONVOLUTION_LAYER, 12, tempConvolutionSize);
    tempCnn->addLayer(SAMPLING_LAYER, -1, tempSampleSize);
    tempCnn->addLayer(OUTPUT_LAYER, tempNumClasses, nullptr);

    //Setup.
    printf("Setup\r\n");
    tempCnn->setup();

    //Step 2. Read data
    printf("Read training data\r\n");
    string tempString = "D:\\C\\cann\\data\\mnist\\train.format";
    char *tempFilename = (char *)tempString.c_str();

    MfDataReader* tempReader = new MfDataReader(tempFilename);
    printf("Done.\r\n");
    tempReader->randomize();
    tempReader->splitInTwo(0.2);

    MfDoubleMatrix* tempTrainingX = tempReader->getTrainingX();
    MfIntArray* tempTrainingY = tempReader->getTrainingY();
    MfDoubleMatrix* tempTestingX = tempReader->getTestingX();
    MfIntArray* tempTestingY = tempReader->getTestingY();

    //printf("Read testing data\r\n");
    //tempString = "D:\\C\\cann\\data\\mnist\\test.format";
    //tempFilename = (char *)tempString.c_str();

    //MfDataReader* tempReader2 = new MfDataReader(tempFilename);
    //tempReader->randomize();
    //tempReader->splitInTwo(0.6);

    //MfDoubleMatrix* tempTestingX = tempReader2->getWholeX();
    //MfIntArray* tempTestingY = tempReader2->getWholeY();
    //printf("Done.\r\n");

    tempCnn->initializeRandomArray(tempTrainingX->getRows());
    printf("Start training rounds:\r\n");
    for(int i = 0; i < 5; i ++)
    {
        printf("Round: %d:\r\n", i);
        tempCnn->train(tempTrainingX, tempTrainingY);
    }//Of for i

    printf("After training:\r\n\r\n\r\n\r\n");
    double tempAccuracy = tempCnn->test(tempTestingX, tempTestingY);

    //free(tempMfIntArray);

    free(tempCnn);

    printf("The accuracy is: %lf. Finish. \r\n", tempAccuracy);
}//Of trainingTestingTest

/**
 * Code unit test.
 */
void MfFullCnn::unitTest()
{
    //Test the digit recognition data.
    MfFullCnn* tempCnn = new MfFullCnn(10);
    MfSize* tempImageSize = new MfSize(28, 28);
    MfSize* tempConvolutionSize = new MfSize(5, 5);
    MfSize* tempSampleSize = new MfSize(2, 2);
    int tempNumClasses = 10;

    //Four layers
    tempCnn->addLayer(INPUT_LAYER, -1, tempImageSize);
    tempCnn->addLayer(CONVOLUTION_LAYER, 6, tempConvolutionSize);
    tempCnn->addLayer(SAMPLING_LAYER, -1, tempSampleSize);
    tempCnn->addLayer(CONVOLUTION_LAYER, 12, tempConvolutionSize);
    tempCnn->addLayer(SAMPLING_LAYER, -1, tempSampleSize);
    tempCnn->addLayer(OUTPUT_LAYER, tempNumClasses, nullptr);

    //Setup.
    tempCnn->setup();

    //Now load the data for training/testing.

}//Of unitTest

