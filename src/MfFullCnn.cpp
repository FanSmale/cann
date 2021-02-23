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
 * The destructor.
 */
MfFullCnn::~MfFullCnn()
{
    //dtor
}//Of the destructor

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
 * Setup.
 */
void MfFullCnn::setup()
{
    for(int i = 0; i < numLayers; i ++)
    {
        layers[i]->setup();
    }//Of for i
}//Of setup

/**
 * Forward computing. Similar to ANN, invoke forward of each layer.
 * paraData: the given training instance.
 * paraLabel: the label of the instance.
 */
int MfFullCnn::forward(MfDoubleMatrix* paraData) {
    //printf("forward:\r\n");
    //cout << paraData->toString() << endl;
    for (int i = 0; i < numLayers; i++)
    {
        //printf("Layer[%d] forward\r\n", i);
        layers[i]->forward(paraData);
    }//Of for i

    int tempPrediction = layers[numLayers - 1]->getCurrentPrediction();

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
 * Return: the prediction accuracy.
 */
double MfFullCnn::train(MfDoubleMatrix* paraX, MfIntArray* paraY)
{
    int tempRows = paraX->getRows();
    int tempColumns = paraX->getColumns();
    int tempEpochs = tempRows / batchSize;

    int tempInstance;
    int tempLabel;
    int tempPrediction;

    double tempCorrect = 0.0;

    MfDoubleMatrix* tempData = new MfDoubleMatrix(1, tempColumns);

    randomize();
    for(int e = 0; e < tempEpochs; e ++)
    {
        //A new batch
        prepareForNewBatch();
        for(int i = 0; i < batchSize; i ++)
        {
            tempInstance = randomArray->getValue(e * batchSize + i);
            for(int j = 0; j < tempColumns; j ++)
            {
                tempData->setValue(0, j, paraX->getValue(tempInstance, j));
            }//Of for j
            tempLabel = paraY->getValue(tempInstance);
            tempPrediction = forward(tempData);
            if (tempPrediction == tempLabel)
            {
                tempCorrect ++;
            }//Of if
            backPropagation(tempLabel);

            //A new record
            prepareForNewRecord();
        }//Of for i

        //Update for each batch
        //printf("\r\n updateParameters\r\n");
        updateParameters();
    }//Of for e

    return tempCorrect/tempRows;
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
    for(int i = 0; i < tempRows; i ++)
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
        }//Of if
    }//Of for i

    outputKernelsToFile();

    return tempNumCorrect/tempRows;
}//Of test

/**
 * Output kernels for debugging.
 */
void MfFullCnn::outputKernelsToFile()
{
    FILE *tempFile;
    if((tempFile = fopen("D:\\C\\cann\\data\\kernels.txt", "w")) == NULL)
    {
        printf("Could not open file for writing.\r\n");
        exit(1);
    }//Of if
    fprintf(tempFile, "These are kernels for a total of %d layers:\r\n", numLayers);
    for(int i = 0; i < numLayers; i ++)
    {
        if((layers[i]->getLayerType() == INPUT_LAYER) ||
           (layers[i]->getLayerType() == SAMPLING_LAYER))
        {
            continue;
        }//Of if

        fprintf(tempFile, "\r\nLayer %d\r\n", i);

        fprintf(tempFile, "%s", layers[i]->getKernel()->toString().c_str());
    }//Of for i
    fprintf(tempFile, "\r\nEnd of file\r\n");
    fclose(tempFile);
}//Of outputKernelsToFile

/**
 * Training/testing test using the digit recognition data.
 */
void MfFullCnn::mnistTest()
{
    //Step 1. Build the CNN
    printf("Building CNN\r\n");
    MfFullCnn* tempCnn = new MfFullCnn(5);
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
    tempReader->splitInTwo(0.8);

    MfDoubleMatrix* tempTrainingX = tempReader->getTrainingX();
    MfIntArray* tempTrainingY = tempReader->getTrainingY();
    MfDoubleMatrix* tempTestingX = tempReader->getTestingX();
    MfIntArray* tempTestingY = tempReader->getTestingY();

    tempCnn->initializeRandomArray(tempTrainingX->getRows());
    printf("Start training rounds:\r\n");
    double tempAccuracy;
    for(int i = 0; i < 10; i ++)
    {
        tempAccuracy = tempCnn->train(tempTrainingX, tempTrainingY);
        printf("Round: %d, accuracy = %lf:\r\n", i, tempAccuracy);
    }//Of for i

    printf("After training:\r\n\r\n\r\n");
    tempCnn->outputKernelsToFile();

    printf("Before testing:\r\n");
    tempAccuracy = tempCnn->test(tempTestingX, tempTestingY);

    free(tempCnn);

    printf("The accuracy is: %lf. Finish. \r\n", tempAccuracy);
}//Of mnistTest

/**
 * Code integrated test.
 */
void MfFullCnn::integratedTest()
{
    //Test the digit recognition data.
    MfFullCnn* tempCnn = new MfFullCnn(1);
    MfSize* tempImageSize = new MfSize(6, 6);
    MfSize* tempConvolutionSize = new MfSize(3, 3);
    MfSize* tempSampleSize = new MfSize(2, 2);
    int tempNumClasses = 2;

    //Four layers
    tempCnn->addLayer(INPUT_LAYER, -1, tempImageSize);
    tempCnn->addLayer(CONVOLUTION_LAYER, 2, tempConvolutionSize);
    tempCnn->addLayer(SAMPLING_LAYER, -1, tempSampleSize);
    tempCnn->addLayer(OUTPUT_LAYER, tempNumClasses, nullptr);

    //Setup.
    tempCnn->setup();

    printf("After setup\r\n");

    //Now load the data for training/testing.
    MfDoubleMatrix* tempTrainingX = new MfDoubleMatrix(1, 36);
    for(int i = 0; i < 18; i ++) {
        tempTrainingX->setValue(0, i, 0);
    }
    for(int i = 18; i < 36; i ++) {
        tempTrainingX->setValue(0, i, 1);
    }
    MfIntArray* tempTrainingY = new MfIntArray(1);
    tempTrainingY->fill(1);
    tempCnn->initializeRandomArray(tempTrainingX->getRows());

    printf("before training\r\n");
    tempCnn->train(tempTrainingX, tempTrainingY);

    printf("After training:\r\n\r\n");
    tempCnn->outputKernelsToFile();

    //double tempAccuracy = tempCnn->test(tempTestingX, tempTestingY);

    free(tempCnn);
}//Of integratedTest

