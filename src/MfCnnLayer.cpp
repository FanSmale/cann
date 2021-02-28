/*
 * The C++ Artificial Neural network project.
 * This class handles CNN layer.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

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

    alpha = 0.85;
    lambda = 0;

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
        predictionDistribution = new MfDoubleMatrix(1, numClasses);
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
    free(predictionDistribution);
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
    //Attention: I do not know why these thresholds work.
    kernel->fill(-0.005, 0.095);
    //kernel->fill(-0.5, 0.5);

    currentKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    singleDeltaKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    deltaKernel = new MfDoubleMatrix(kernelSize->width, kernelSize->height);
    currentRot180Kernel = new MfDoubleMatrix(kernelSize->height, kernelSize->width);
}//Of initKernel

/**
 * Initialize the bias.
 */
void MfCnnLayer::initBias()
{
    bias = new MfDoubleMatrix(1, numOutMaps);
    bias->fill(0);
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
 * Get the kernel at the given position.
 * paraFrontMap: the front map number.
 * paraOutMap: the out map number.
 */
MfDoubleMatrix* MfCnnLayer::getKernelAt(int paraFrontMap, int paraOutMap)
{
    double** tempMatrix = kernel->getData()[paraFrontMap][paraOutMap];
    double** tempNewMatrix = currentKernel->getData();
    for (int i = 0; i < currentKernel->getRows(); i++)
    {
        for (int j = 0; j < currentKernel->getColumns(); j++)
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
    for (int i = 0; i < currentKernel->getRows(); i++)
    {
        for (int j = 0; j < currentKernel->getColumns(); j++)
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
    for (int i = 0; i < paraKernel->getRows(); i++)
    {
        for (int j = 0; j < paraKernel->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i
}//Of setKernelAt

/**
 * Get the map at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getOutMapAt(int paraOutMapNo)
{
    return getOutMapAt(recordInBatch, paraOutMapNo);
}//Of getOutMapAt

/**
 * Get the map at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getOutMapAt(int paraRecordInBatch, int paraOutMapNo)
{
    double** tempMatrix = outMaps->getData()[paraRecordInBatch][paraOutMapNo];
    double** tempNewMatrix = currentOutMap->getData();
    for (int i = 0; i < currentOutMap->getRows(); i++)
    {
        for (int j = 0; j < currentOutMap->getColumns(); j++)
        {
            tempNewMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i

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

    for (int i = 0; i < tempX; i ++)
    {
        for (int j = 0; j < tempY; j ++)
        {
            newMatrix[i][j] = tempMatrix[i][j];
        }//Of for j
    }//Of for i
}//Of setOutMapValue

/**
 * Get errors at the given position.
 */
MfDoubleMatrix* MfCnnLayer::getErrorsAt(int paraMapNo)
{
    double** tempMatrix = errors->getData()[recordInBatch][paraMapNo];
    double** tempNewMatrix = currentErrors->getData();
    for (int i = 0; i < currentErrors->getRows(); i++)
    {
        for (int j = 0; j < currentErrors->getColumns(); j++)
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
    for (int i = 0; i < paraMatrix->getRows(); i++)
    {
        for (int j = 0; j < paraMatrix->getColumns(); j++)
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
    for (int i = 0; i < currentErrors->getRows(); i++)
    {
        for (int j = 0; j < currentErrors->getColumns(); j++)
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
        printf("For INPUT_LAYER ...");
        initOutMaps();
        printf("done.\r\n");
        break;
    case CONVOLUTION_LAYER:
        printf("For CONVOLUTION_LAYER ...");
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
        getMapSize()->divideToMe(lastLayer->getMapSize(), getScaleSize());
        initErrors();
        initOutMaps();
        printf("done.\r\n");
        break;
    case OUTPUT_LAYER:
        printf("For OUTPUT_LAYER ...");
        kernelSize->cloneToMe(lastLayer->getMapSize());
        initKernel(tempNumFrontMaps);
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
            setOutMapValue(0, i, j, paraData->getValue(0, mapSize->height * i + j));
        }//Of for j
    }//Of for i
}//Of setInputLayerOutput

/**
 * Compute the convolution output according to the output of the last layer.
 * paraLastLayer: the last layer.
 * paraLayer: the current layer.
 */
void MfCnnLayer::setConvolutionOutput()
{
    int tempLastNumMaps = lastLayer->getNumOutMaps();
    MfDoubleMatrix* tempMap;
    MfDoubleMatrix* tempKernel;
    double tempBias;

    for (int j = 0; j < numOutMaps; j++)
    {
        currentOutMap->fill(0);
        for (int i = 0; i < tempLastNumMaps; i++)
        {
            tempMap = lastLayer->getOutMapAt(i);
            tempKernel = getKernelAt(i, j);
            //Sum up convolution maps
            singleOutMap->convolutionValidToMe(tempMap, tempKernel);
            currentOutMap->addToMe(currentOutMap, singleOutMap);
        }//Of for i

        //Bias.
        tempBias = getBiasAt(j);
        currentOutMap->addValueToMe(tempBias);

        //Activation.
        currentOutMap->setActivator(layerActivator);
        currentOutMap->activate();

        setOutMapValue(j, currentOutMap);
    }//Of for j
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
    double tempValue;
    int resultPrediction = -1;
    double tempMaxValue = -1000;

    for (int i = 0; i < numClasses; i++)
    {
        tempValue = getOutMapAt(i)->getValue(0, 0);
        predictionDistribution->setValue(0, i, tempValue);
        if (tempMaxValue < tempValue)
        {
            tempMaxValue = tempValue;
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

    MfDoubleMatrix* tempNextErrors;
    MfDoubleMatrix* tempRot180Kernel;

    for (int i = 0; i < numOutMaps; i++)
    {
        currentErrors->fill(0);
        for (int j = 0; j < tempNextMapNum; j++) {
            tempNextErrors = nextLayer->getErrorsAt(j);
            tempRot180Kernel = nextLayer->getRot180KernelAt(i, j);

            currentSingleErrors->convolutionFullToMe(tempNextErrors, tempRot180Kernel);
            currentErrors->addToMe(currentErrors, currentSingleErrors);
        }//Of for j

        setErrorsAt(i, currentErrors);
    }//Of for i
}//Of setSamplingLayerErrors

/**
 * Set the output layer errors.
 * paraData: the instance.
 * paraLabel: the actual label of the instance.
 */
void MfCnnLayer::setOutputLayerErrors(int paraLabel)
{
    double* tempTarget = (double*)malloc(numOutMaps * sizeof(double));
    double* tempOutmaps = (double*)malloc(numOutMaps * sizeof(double));
    double tempValue;

    for (int i = 0; i < numOutMaps; i++)
    {
        tempTarget[i] = 0;
        tempOutmaps[i] = getOutMapAt(i)->getValue(0, 0);
    }//Of for i

    tempTarget[paraLabel] = 1;

    for (int i = 0; i < numOutMaps; i ++)
    {
        tempValue = layerActivator->derive(tempOutmaps[i]) * (tempTarget[i] - tempOutmaps[i]);
        setErrorAt(i, 0, 0, tempValue);
    }//Of for i

    free(tempTarget);
    free(tempOutmaps);
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
}//Of backPropagation

/**
 * Update kernels.
 */
void MfCnnLayer::updateKernels()
{
    int tempNumLastMap = lastLayer->getNumOutMaps();

    for (int j = 0; j < numOutMaps; j++)
    {
        for (int i = 0; i < tempNumLastMap; i++)
        {
            //tempFirst = true;
            deltaKernel->fill(0);
            for (int r = 0; r < batchSize; r++)
            {
                currentErrors = getErrorsAt(r, j);

                singleDeltaKernel->convolutionValidToMe(lastLayer->getOutMapAt(r, i), currentErrors);
                deltaKernel->addToMe(deltaKernel, singleDeltaKernel);
            }//Of for r

            currentKernel = getKernelAt(i, j);
            currentKernel->timesValueToMe(1 - lambda * alpha);
            currentKernel->addToMe(currentKernel, deltaKernel);

            setKernelAt(i, j, currentKernel);
        }//Of for i
    }//Of for j
}//Of updateKernels

/**
 * Update bias.
 */
void MfCnnLayer::updateBias() {
    double tempBias;
    double tempDeltaBias;

    for (int j = 0; j < numOutMaps; j ++)
    {
        errors->sumToMatrix(j, currentErrors);
        tempDeltaBias = currentErrors->sumUp() / batchSize;

        tempBias = getBiasAt(j) + alpha * tempDeltaBias;
        setBiasAt(j, tempBias);
    }//Of for i
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
 * Code unit test.
 */
void MfCnnLayer::unitTest()
{

}//Of unitTest


