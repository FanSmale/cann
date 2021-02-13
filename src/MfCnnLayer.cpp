#include "MfCnnLayer.h"

/**
 * The default constructor.
 */
MfCnnLayer::MfCnnLayer()
{
}//Of the destructor

/**
 * The second constructor.
 * paraLayerType: the type of the layer.
 * paraBatchSize: the batch size, for batch training.
 * paraNum: the given number, the use differs for different types of layers.
 * paraSize: the given size, the use differs for different types of layers.
 */
MfCnnLayer::MfCnnLayer(int paraLayerType, int paraBatchSize, int paraNum, MfSize* paraSize)
{
    //Accept parameter
    layerType = paraLayerType;
    batchSize = paraBatchSize;

    //Initialize
    mapSize = new MfSize();
    kernelSize = new MfSize();
    scaleSize = new MfSize();
    kernel = nullptr;
    bias = nullptr;
    outMaps = nullptr;
    errors = nullptr;
    layerActivator = nullptr;

    lastLayer = nullptr;
    nextLayer = nullptr;

    switch (layerType)
    {
    case INPUT_LAYER:
        numOutMaps = 1;
        mapSize->cloneToMe(paraSize);
        break;
    case CONVOLUTION_LAYER:
        numOutMaps = paraNum;
        kernelSize->cloneToMe(paraSize);
        break;
    case SAMPLING_LAYER:
        scaleSize->cloneToMe(paraSize);
        break;
    case OUTPUT_LAYER:
        numClasses = paraNum;
        mapSize->setValues(1, 1);
        numOutMaps = numClasses;
        break;
    default:
        printf("Internal error occurred in MfCnnLayer.cpp constructor.\r\n");
        throw "Internal error occurred in MfCnnLayer.cpp constructor.\r\n";
    }// Of switch
}//Of the second constructor

MfCnnLayer::~MfCnnLayer()
{
    free(mapSize);
    free(kernelSize);
    free(scaleSize);
    free(kernel);
    free(currentKernel);
    free(singleDeltaKernel);
    free(deltaKernel);
    free(currentRot180Kernel);
    free(bias);
    free(outMaps);
    free(singleOutMap);
    free(currentOutMap);
    free(errors);
    free(currentErrors);
    free(currentSingleErrors);
}//Of the destructor

/**
 * Initialize the kernel.
 * paraNumFrontMaps: the number of front maps.
 */
void MfCnnLayer::initKernel(int paraNumFrontMaps)
{
    kernel = new Mf4DTensor(paraNumFrontMaps, numOutMaps, kernelSize->width, kernelSize->height);
    currentKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    singleDeltaKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    deltaKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    currentRot180Kernel = new MfDoubleMatrix(kernelSize->height, kernelSize->width);
}//Of initKernel

/**
 * Initialize the output kernel. The essential difference from initKernel is unknown.
 * paraNumFrontMaps: the number of front maps.
 * paraSize: the new size of the kernel (the last two dimensions).
 */
void MfCnnLayer::initOutKernel(int paraNumFrontMaps, MfSize* paraSize)
{
    kernelSize->cloneToMe(paraSize);
    initKernel(paraNumFrontMaps);
}//Of initOutKernel

/**
 * Initialize the bias.
 */
void MfCnnLayer::initBias()
{
    bias = new MfDoubleMatrix(1, numOutMaps);
}//Of initBias

/**
 * Initialize the errors. The first dimension is the batch size instead of numFrontMaps.
 * paraBatchSize: the batch size.
 */
void MfCnnLayer::initErrors()
{
    errors = new Mf4DTensor(batchSize, numOutMaps, mapSize->width, mapSize->height);
    currentSingleErrors = new MfDoubleMatrix(mapSize->width, mapSize->height);
    currentErrors = new MfDoubleMatrix(mapSize->width, mapSize->height);
}//Of initErrors

/**
 * Initialize the output maps. The first dimension is the batch size instead of numFrontMaps.
 * paraBatchSize: the batch size.
 */
void MfCnnLayer::initOutMaps()
{
    outMaps = new Mf4DTensor(batchSize, numOutMaps, mapSize->width, mapSize->height);
    currentOutMap = new MfDoubleMatrix(mapSize->width, mapSize->height);
    singleOutMap = new MfDoubleMatrix(mapSize->width, mapSize->height);
}//Of initOutMaps

/**
 * Prepare for a new batch.
 */
void MfCnnLayer::prepareForNewBatch()
{
    recordInBatch = 0;
}//Of prepareForNewBatch

/**
 * Prepare for a new record.
 */
void MfCnnLayer::prepareForNewRecord()
{
    recordInBatch ++;
}//Of prepareForNewRecord

/**
 * Getter.
 */
int MfCnnLayer::getNumClasses()
{
    return numClasses;
}//Of getNumClasses

/**
 * Getter.
 */
int MfCnnLayer::getLayerType()
{
    return layerType;
}//Of getLayerType

/**
 * Getter.
 */
int MfCnnLayer::getNumOutMaps()
{
    return numOutMaps;
}//Of getNumOutMaps

/**
 * Setter.
 */
void MfCnnLayer::setNumOutMaps(int paraNumOutMaps)
{
    numOutMaps = paraNumOutMaps;
}//Of setNumOutMaps

/**
 * Getter.
 */
MfSize* MfCnnLayer::getMapSize()
{
    return mapSize;
}//Of getMapSize

/**
 * Setter.
 */
void MfCnnLayer::setMapSize(MfSize* paraSize)
{
    mapSize->cloneToMe(paraSize);
}//Of getMapSize

/**
 * Getter.
 */
MfSize* MfCnnLayer::getKernelSize()
{
    return kernelSize;
}//Of getKernelSize

/**
 * Setter.
 */
void MfCnnLayer::setKernelSize(MfSize* paraSize)
{
    kernelSize->cloneToMe(paraSize);
}//Of setKernelSize

/**
 * Getter.
 */
MfSize* MfCnnLayer::getScaleSize()
{
    return scaleSize;
}//Of getScaleSize

/**
 * Setter.
 */
void MfCnnLayer::setScaleSize(MfSize* paraSize)
{
    scaleSize->cloneToMe(paraSize);
}//Of setScaleSize

/**
 * Get the kernel at the given position.
 * paraFrontMap: the front map number.
 * paraOutMap: the out map number.
 */
MfDoubleMatrix* MfCnnLayer::getKernelAt(int paraFrontMap, int paraOutMap)
{
    double** tempMatrix = kernel->getData()[paraFrontMap][paraOutMap];
    double** tempNewMatrix = currentKernel->getData();
    for(int i = 0; i < currentKernel->getRows(); i++)
    {
        for(int j = 0; j < currentKernel->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    return currentKernel;
}//Of getKernelAt

/**
 * Get the rotate 180 kernel at the given position.
 * paraFrontMap: the front map number.
 * paraOutMap: the out map number.
 */
MfDoubleMatrix* MfCnnLayer::getRot180KernelAt(int paraFrontMap, int paraOutMap)
{
    double** tempMatrix = kernel->getData()[paraFrontMap][paraOutMap];
    double** tempNewMatrix = currentKernel->getData();
    for(int i = 0; i < currentKernel->getRows(); i++)
    {
        for(int j = 0; j < currentKernel->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    currentRot180Kernel->rotate180ToMe(currentKernel);
    return currentRot180Kernel;
}//Of getRot180KernelAt

/**
 * Set the kernel at the given position.
 * paraFrontMap: the front map number.
 * paraOutMap: the out map number.
 * paraKernel: the new kernel.
 */
void MfCnnLayer::setKernelAt(int paraFrontMap, int paraOutMap, MfDoubleMatrix* paraKernel)
{
    double** tempMatrix = paraKernel->getData();
    double** tempNewMatrix = kernel->getData()[paraFrontMap][paraOutMap];
    for(int i = 0; i < paraKernel->getRows(); i++)
    {
        for(int j = 0; j < paraKernel->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i
}//Of setKernelAt

/**
 * Get the bias at the given position.
 * paraMapNo: the map number.
 */
double MfCnnLayer::getBiasAt(int paraMapNo)
{
    return bias->getValue(0, paraMapNo);
}//Of getBiasAt

/**
 * Set the bias at the given position.
 * paraMapNo: the map number.
 * paraValue: the given value.
 */
void MfCnnLayer::setBiasAt(int paraMapNo, double paraValue)
{
    bias->setValue(0, paraMapNo, paraValue);
}//Of setBiasAt

/**
 * Get whole outMaps.
 */
Mf4DTensor* MfCnnLayer::getOutMaps()
{
    return outMaps;
}//Of getOutMaps

/**
 * Get the map at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getOutMapAt(int paraOutMapNo)
{
    double** tempMatrix = outMaps->getData()[recordInBatch][paraOutMapNo];
    double** tempNewMatrix = currentOutMap->getData();
    for(int i = 0; i < currentOutMap->getRows(); i++)
    {
        for(int j = 0; j < currentOutMap->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i
    //printf("End of getOutMapAt(). currentOutMap %d * %d: \r\n",
    //       currentOutMap->getRows(), currentOutMap->getColumns());
    return currentOutMap;
}//Of getOutMapAt

/**
 * Get the map at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getOutMapAt(int paraRecordInBatch, int paraOutMapNo)
{
    //printf("Begin of getOutMapAt().\r\n");
    double** tempMatrix = outMaps->getData()[paraRecordInBatch][paraOutMapNo];
    double** tempNewMatrix = currentOutMap->getData();
    for(int i = 0; i < currentOutMap->getRows(); i++)
    {
        for(int j = 0; j < currentOutMap->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    //printf("End of getOutMapAt(). currentOutMap %d * %d: \r\n",
    //       currentOutMap->getRows(), currentOutMap->getColumns());
    return currentOutMap;
}//Of getOutMapAt

/**
 * Set one value of outMaps.
 * paraMapNo: the output map number.
 * paraX: the X index.
 * paraY: the Y index.
 * paraValue: the given value.
 */
void MfCnnLayer::setOutMapValue(int paraMapNo, int paraX, int paraY, double paraValue)
{
    outMaps->setValue(recordInBatch, paraMapNo, paraX, paraY, paraValue);
}// Of setOutMapValue

/**
 * Set one part of outMaps.
 * paraMapNo: the output map number.
 * paraMatrix: the given matrix.
 */
void MfCnnLayer::setOutMapValue(int paraMapNo, MfDoubleMatrix* paraMatrix)
{
    double** tempMatrix = paraMatrix->getData();
    int tempX = paraMatrix->getRows();
    int tempY = paraMatrix->getColumns();
    double** newMatrix = outMaps->getData()[recordInBatch][paraMapNo];

    for(int i = 0; i < tempX; i ++)
    {
        for(int j = 0; j < tempY; j ++)
        {
            newMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    //printf("The out maps is:\r\n");
    //cout << outMaps->toString() <<endl;
}//Of setOutMapValue

/**
 * Get whole errors.
 */
Mf4DTensor* MfCnnLayer::getErrors()
{
    return errors;
}//Of getErrors

/**
 * Get errors at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getErrorsAt(int paraMapNo)
{
    double** tempMatrix = errors->getData()[recordInBatch][paraMapNo];
    double** tempNewMatrix = currentErrors->getData();
    for(int i = 0; i < currentErrors->getRows(); i++)
    {
        for(int j = 0; j < currentErrors->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    return currentErrors;
}//Of getErrorsAt

/**
 * Set errors at the given position.
 */
void MfCnnLayer::setErrorsAt(int paraMapNo, MfDoubleMatrix* paraMatrix)
{
    double** tempMatrix = paraMatrix->getData();
    double** tempNewMatrix = errors->getData()[recordInBatch][paraMapNo];
    for(int i = 0; i < paraMatrix->getRows(); i++)
    {
        for(int j = 0; j < paraMatrix->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i
}//Of setErrorsAt

/**
 * Get errors at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getErrorsAt(int paraRecordInBatch, int paraMapNo)
{
    double** tempMatrix = errors->getData()[paraRecordInBatch][paraMapNo];
    double** tempNewMatrix = currentErrors->getData();
    for(int i = 0; i < currentErrors->getRows(); i++)
    {
        for(int j = 0; j < currentErrors->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

    return currentErrors;
}//Of getErrorsAt

/**
 * Get error at the given position.
 */
void MfCnnLayer::setErrorAt(int paraMapNo, int paraMapX, int paraMapY, double paraValue)
{
    errors->setValue(recordInBatch, paraMapNo, paraMapX, paraMapY, paraValue);
}//Of getErrorAt

/**
 * Setter.
 */
void MfCnnLayer::setLastLayer(MfCnnLayer* paraLayer)
{
    lastLayer = paraLayer;
}//Of setLastLayer

/**
 * Setter.
 */
void MfCnnLayer::setNextLayer(MfCnnLayer* paraLayer)
{
    nextLayer = paraLayer;
}//Of setLastLayer

/**
 * Setup.
 */
void MfCnnLayer::setup()
{
    int tempNumFrontMaps = 0;
    if (lastLayer != nullptr)
    {
        tempNumFrontMaps = lastLayer->getNumOutMaps();
    }//Of if
    switch (layerType)
    {
    case INPUT_LAYER:
        printf("initOutMaps ...");
        initOutMaps();
        printf("done.\r\n");
        break;
    case CONVOLUTION_LAYER:
        printf("For CONVOLUTION_LAYER ...");
        //layers[i].setMapSize(layers[i - 1].getMapSize().subtract(layers[i].getKernelSize(), 1));
        //Output map size is determined by the map size of this layer and the kernel size.
        getMapSize()->subtractToMe(lastLayer->getMapSize(), kernelSize, 1);
        initKernel(tempNumFrontMaps);
        initBias();
        initErrors();
        initOutMaps();
        printf("done.\r\n");
        break;
    case SAMPLING_LAYER:
        printf("For SAMPLING_LAYER ...");
        setNumOutMaps(tempNumFrontMaps);
        //layers[i].setMapSize(layers[i - 1].getMapSize().divide(layers[i].getScaleSize()));
        getMapSize()->divideToMe(lastLayer->getMapSize(), getScaleSize());
        initErrors();
        initOutMaps();
        printf("done.\r\n");
        break;
    case OUTPUT_LAYER:
        printf("For OUTPUT_LAYER ...");
        initOutKernel(tempNumFrontMaps, lastLayer->getMapSize());
        initBias();
        initErrors();
        initOutMaps();
        printf("done.\r\n");
        break;
    }//Of switch
}//Of setup

/**
 * Set the in layer output.
 * Given a record, copy its values to the input map.
 */
void MfCnnLayer::setInputLayerOutput(MfDoubleMatrix* paraData)
{
    //printf("MfCnnLayer::setInputLayerOutput, recordInBatch = %d\r\n", recordInBatch);
    if (paraData->getColumns() != mapSize->width * mapSize->height)
    {
        printf("input record does not match the map size.\r\n");
        throw "input record does not match the map size.";
    }//Of if

    for (int i = 0; i < mapSize->width; i++)
    {
        for (int j = 0; j < mapSize->height; j++)
        {
            //The input layer has only 1 out map.
            //inputLayer.setMapValue(0, i, j, attr[mapSize.y * i + j]);
            setOutMapValue(0, i, j, paraData->getValue(0, mapSize->height * i + j));
        }//Of for j
    }//Of for i
    //printf("MfCnnLayer::setInputLayerOutput end\r\n");
}//Of setInputLayerOutput

/**
 * Compute the convolution output according to the output of the last layer.
 * paraLastLayer: the last layer.
 * paraLayer: the current layer.
 */
void MfCnnLayer::setConvolutionOutput()
{
    //printf("MfCnnLayer::setConvolutionOutput()\r\n");
    int tempLastNumMaps = lastLayer->getNumOutMaps();
    MfDoubleMatrix* tempMap;
    MfDoubleMatrix* tempKernel;
    double tempBias;
    bool tempEmpty = true;

    /////////////////For test output begins
    //tempMap = lastLayer->getOutMapAt(0);
    //printf("last layer map is %d * %d: \r\n", tempMap->getRows(), tempMap->getColumns());
    //cout << tempMap->toString() << endl;
    //tempKernel = getKernelAt(0, 0);
    //printf("last layer kernel is %d * %d: \r\n", tempKernel->getRows(), tempKernel->getColumns());
    //cout << tempKernel->toString() << endl;

    //printf("singleOutMap %d * %d: \r\n", singleOutMap->getRows(), singleOutMap->getColumns());
    //cout << singleOutMap->toString() << endl;
    //printf("currentOutMap %d * %d: \r\n", currentOutMap->getRows(), currentOutMap->getColumns());
    //cout << currentOutMap->toString() << endl;
    /////////////////For test output ends

    for (int j = 0; j < numOutMaps; j++)
    {
        tempEmpty = true;
        for (int i = 0; i < tempLastNumMaps; i++)
        {
            tempMap = lastLayer->getOutMapAt(i);
            tempKernel = getKernelAt(i, j);
            if (tempEmpty)
            {
                //Only convolution on one map.
                currentOutMap->convolutionValidToMe(tempMap, tempKernel);
            }
            else
            {
                //Sum up convolution maps
                currentOutMap->addToMe(currentOutMap, singleOutMap->convolutionValidToMe(tempMap, tempKernel));
            }//Of if
        }//Of for i

        //Bias.
        tempBias = getBiasAt(j);
        currentOutMap->addValueToMe(tempBias);

        //Activation.
        currentOutMap->setActivator(layerActivator);
        currentOutMap->activate();
        //printf("currentOutMap is: \r\n");
        //cout << currentOutMap->toString() << endl;

        setOutMapValue(j, currentOutMap);
    }//Of for j
    //printf("End of setConvolutionOutput(). currentOutMap %d * %d: \r\n",
    //       currentOutMap->getRows(), currentOutMap->getColumns());
}//Of setConvolutionOutput

/**
 * Compute the sampling output. It is the averaging of scaleSize.
 * The number of output maps is the same as the last layer.
 */
void MfCnnLayer::setSamplingOutput()
{
    int tempLastMapNum = lastLayer->getNumOutMaps();

    for (int i = 0; i < tempLastMapNum; i++) {
        currentOutMap->scaleToMe(lastLayer->getOutMapAt(i), scaleSize);
        setOutMapValue(i, currentOutMap);
    }//Of for i
}//Of setSamplingOutput

/**
 * Get the current prediction. Only valid for the last layer.
 * Return: the current prediction.
 */
int MfCnnLayer::getCurrentPrediction()
{
    double tempOutmaps[numOutMaps];
    int resultPrediction = -1;
    double tempMaxValue = -1000;

    for (int i = 0; i < numOutMaps; i++)
    {
        //outmaps->setValue(0, i, getMapAt(i)->getValue(0, 0));
        tempOutmaps[i] = getOutMapAt(i)->getValue(0, 0);
        if (tempMaxValue < tempOutmaps[i])
        {
            tempMaxValue = tempOutmaps[i];
            resultPrediction = i;
        }//Of if
    }//Of for i

    return resultPrediction;
}//Of getCurrentPrediction

/**
 * Forward according to the layer type.
 */
void MfCnnLayer::forward(MfDoubleMatrix* paraData)
{
    switch (layerType)
    {
    case INPUT_LAYER:
        setInputLayerOutput(paraData);
        break;
    case CONVOLUTION_LAYER:
        setConvolutionOutput();
        break;
    case SAMPLING_LAYER:
        setSamplingOutput();
        break;
    case OUTPUT_LAYER:
        setConvolutionOutput();
        break;
    default:
        printf("Unsupported layer type.\r\n");
        throw "Unsupported layer type.\r\n";
        break;
    }//Of switch

    //printf("End of MfCnnLayer::forward(), currentOutMap %d * %d: \r\n",
    //       currentOutMap->getRows(), currentOutMap->getColumns());
}//Of forward

/**
 * Set convolution layer errors for back propagation.
 */
void MfCnnLayer::setConvolutionLayerErrors()
{
    for (int i = 0; i < numOutMaps; i ++)
    {
        currentOutMap = getOutMapAt(i);
        currentOutMap->setActivator(layerActivator);
        currentOutMap->deriveToMe(currentOutMap);

        /*
        if (!currentOutMap->rangeCheck(-5, 5))
        {
            printf("MfCnnLayer::setConvolutionLayerErrors, a value of currentOutMap exceeds [-5, 5].\r\n");
            cout << layerActivator->toString() << endl;
            cout << currentOutMap->toString() << endl;
            throw "MfCnnLayer::setConvolutionLayerErrors";
        }//Of if
        */

        //The space of singleOutMap is reused here, in fact here is the error.
        singleOutMap->kroneckerToMe(nextLayer->getErrorsAt(i), nextLayer->getScaleSize());
        currentOutMap->cwiseProductToMe(currentOutMap, singleOutMap);
        setErrorsAt(i, currentOutMap);
    }//Of for i
}//Of setConvolutionLayerErrors

/**
 * Set the sample layer errors.
 * paraData: the instance.
 * paraLabel: the actual label of the instance.
 * Return: whether or not the current prediction is correct.
 */
void MfCnnLayer::setSamplingLayerErrors()
{
    int tempNextMapNum = nextLayer->getNumOutMaps();

    bool tempFirst;
    MfDoubleMatrix* tempNextErrors;
    MfDoubleMatrix* tempRot180Kernel;
    for (int i = 0; i < numOutMaps; i++)
    {
        tempFirst = true;
        for (int j = 0; j < tempNextMapNum; j++) {
            tempNextErrors = nextLayer->getErrorsAt(j);
            tempRot180Kernel = nextLayer->getRot180KernelAt(i, j);
            if (tempFirst)
            {
                currentErrors->convolutionFullToMe(tempNextErrors, tempRot180Kernel);
                tempFirst = false;
            }
            else
            {
                currentSingleErrors->convolutionFullToMe(tempNextErrors, tempRot180Kernel);
                currentErrors->addToMe(currentErrors, currentSingleErrors);
            }//Of if
        }//Of for j
        setErrorsAt(i, currentErrors);
    }//Of for i
}//Of setSamplingLayerErrors

/**
 * Set the output layer errors.
 * paraData: the instance.
 * paraLabel: the actual label of the instance.
 * Return: whether or not the current prediction is correct.
 */
void MfCnnLayer::setOutputLayerErrors(int paraLabel)
{
    double tempTarget[numOutMaps];
    double tempOutmaps[numOutMaps];

    //MfDoubleMatrix* target = new MfDoubleMatrix(1, numOutMaps);
    //target->fill(0);
    //MfDoubleMatrix* outmaps = new MfDoubleMatrix(1, numOutMaps);
    for (int i = 0; i < numOutMaps; i++)
    {
        tempTarget[i] = 0;
        tempOutmaps[i] = getOutMapAt(i)->getValue(0, 0);
    }//Of for i

    //target->setValue(0, paraLabel, 1);
    tempTarget[paraLabel] = 1;

    for (int i = 0; i < numOutMaps; i ++)
    {
        setErrorAt(i, 0, 0, tempOutmaps[i] * (1 - tempOutmaps[i]) * (tempTarget[i] - tempOutmaps[i]));
    }//Of for i
}//Of setOutputLayerErrors

/**
 * Back propagation according to the layer type.
 */
void MfCnnLayer::backPropagation(int paraLabel)
{
    switch (layerType)
    {
    case INPUT_LAYER:
        printf("Input layer should not back propagation.\r\n");
        throw "Input layer should not back propagation.";
        break;
    case CONVOLUTION_LAYER:
        setConvolutionLayerErrors();
        break;
    case SAMPLING_LAYER:
        setSamplingLayerErrors();
        break;
    case OUTPUT_LAYER:
        setOutputLayerErrors(paraLabel);
        break;
    default:
        printf("Unsupported layer type.\r\n");
        throw "Unsupported layer type.\r\n";
        break;
    }//Of switch

    //printf("End of MfCnnLayer::backPropagation()");
}//Of backPropagation

/**
 * Update kernels.
 */
void MfCnnLayer::updateKernels()
{
    //printf("updateKernels\r\n");
    int tempNumLastMap = lastLayer->getNumOutMaps();
    bool tempFirst = true;

    for (int j = 0; j < numOutMaps; j++)
    {
        //printf("j = %d\r\n", j);
        for (int i = 0; i < tempNumLastMap; i++)
        {
            //printf("i = %d\r\n", i);
            tempFirst = true;
            for (int r = 0; r < batchSize; r++)
            {
                currentErrors = getErrorsAt(r, j);
                //printf("errors got\r\n");
                if (tempFirst)
                {
                    tempFirst = false;
                    deltaKernel->convolutionValidToMe(lastLayer->getOutMapAt(r, i), currentErrors);
                }
                else
                {
                    singleDeltaKernel->convolutionValidToMe(lastLayer->getOutMapAt(r, i), currentErrors);
                    deltaKernel->addToMe(deltaKernel, singleDeltaKernel);
                }//Of if
            }//Of for r

            double tempValue;
            //printf("deltaKernel\r\n");
            deltaKernel->timesValueToMe(ALPHA/batchSize);
            for(int ki = 0; ki < deltaKernel->getRows(); ki ++)
            {
                for(int kj = 0; kj < deltaKernel->getColumns(); kj ++)
                {
                    tempValue = deltaKernel->getValue(ki, kj);
                    if (tempValue < -2 || tempValue > 2)
                    {
                        printf("A strange delta kernel value: %lf\r\n", tempValue);
                        exit(1);
                    }
                }
            }

            currentKernel = getKernelAt(i, j);
            //currentKernel->timesValueToMe(1 - LAMBDA * ALPHA);
            currentKernel->timesValueToMe(1 - ALPHA);
            currentKernel->addToMe(currentKernel, deltaKernel);

            /*
            for(int ki = 0; ki < currentKernel->getRows(); ki ++)
            {
                for(int kj = 0; kj < currentKernel->getColumns(); kj ++)
                {
                    tempValue = currentKernel->getValue(ki, kj);
                    if (tempValue < -5 || tempValue > 5)
                    {
                        printf("A strange kernel value: %lf\r\n", tempValue);

                        printf("kernel = %s\r\n", currentKernel->toString().c_str());

                        printf("deltaKernel = %s\r\n", deltaKernel->toString().c_str());
                        exit(1);
                    }
                }
            }
            */
            setKernelAt(i, j, currentKernel);
        }//Of for i
    }//Of for j

    //printf("The new kernel is: \r\n");
    //cout<< kernel->toString() << endl;
}//Of updateKernels

/**
 * Update bias.
 */
void MfCnnLayer::updateBias() {
    double tempBias;
    double tempDeltaBias;
    double tempArea = currentErrors->getRows() * currentErrors->getColumns();

    //printf("The old bias is: \r\n");
    //cout<< bias->toString() << endl;

    for (int j = 0; j < numOutMaps; j ++)
    {
        errors->sumToMatrix(j, currentErrors);
        tempDeltaBias = currentErrors->sumUp() / batchSize / tempArea;
        if (tempDeltaBias < -5 || tempDeltaBias > 5)
        {
            printf("tempDeltaBias = %lf\r\n", tempDeltaBias);
            FILE *tempFile;
            if ((tempFile = fopen("D:\\C\\cann\\data\\tempoutput.txt", "w")) == NULL)
            {
                printf("Could not open file for writing.\r\n");
                exit(1);
            }//Of if
            fprintf(tempFile, "The deltaBias is too big\r\n");
            fprintf(tempFile, "%s", errors->toString().c_str());
            fprintf(tempFile, "\r\nCurrent errors\r\n");
            fprintf(tempFile, "%s", currentErrors->toString().c_str());
            fprintf(tempFile, "End of file\r\n");
            fclose(tempFile);
            throw "deltaBias too big.";
        }
        tempBias = getBiasAt(j) + ALPHA * tempDeltaBias;
        setBiasAt(j, tempBias);
    }//Of for i

    //printf("The new bias is: \r\n");
    //cout<< bias->toString() << endl;
}//Of updateBias

/**
 * Set the layer activator.
 */
void MfCnnLayer::setLayerActivator(char paraFunction)
{
    if (layerActivator != nullptr)
    {
        layerActivator->setActivationFunction(paraFunction);
    }
    else
    {
        layerActivator = new Activator(paraFunction);
    }//Of if
}//Of setLayerActivator

/**
 * Getter.
 */
MfDoubleMatrix* MfCnnLayer::getCurrentOutMap()
{
    return currentOutMap;
}//Of getCurrentOutMap

/**
 * Code unit test.
 */
void MfCnnLayer::unitTest()
{

}//Of unitTest


