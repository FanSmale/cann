#include "MfCnnLayer.h"

/**
 * The default constructor.
 */
MfCnnLayer::MfCnnLayer()
{
    //ctor
}

/**
 * The second constructor.
 * paraLayerType: the type of the layer.
 * paraNum: the given number, the use differs for different types of layers.
 * paraSize: the given size, the use differs for different types of layers.
 */
MfCnnLayer::MfCnnLayer(int paraLayerType, int paraNum, MfSize* paraSize)
{
    //Accept parameter
    layerType = paraLayerType;

    //Initialize
    mapSize = new MfSize();
    kernelSize = new MfSize();
    scaleSize = new MfSize();
    kernel = nullptr;
    bias = nullptr;
    outMaps = nullptr;
    errors = nullptr;

    switch (layerType) {
    case INPUT_LAYER:
        numOutputMaps = 1;
        mapSize->cloneToMe(paraSize);
        break;
    case CONVOLUTION_LAYER:
        numOutputMaps = paraNum;
        kernelSize->cloneToMe(paraSize);
        break;
    case SAMPLING_LAYER:
        scaleSize->cloneToMe(paraSize);
        break;
    case OUTPUT_LAYER:
        numClasses = paraNum;
        mapSize->setValues(1, 1);
        numOutputMaps = numClasses;
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
    free(bias);
    free(outMaps);
    free(errors);
}//Of the destructor

/**
 * Initialize the kernel.
 * paraNumFrontMaps: the number of front maps.
 */
void MfCnnLayer::initKernel(int paraNumFrontMaps)
{
    kernel = new Mf4DTensor(paraNumFrontMaps, numOutputMaps, kernelSize->width, kernelSize->height);
}//Of initKernel

/**
 * Initialize the output kernel. The essential difference from initKernel is unknown.
 * paraNumFrontMaps: the number of front maps.
 * paraSize: the new size of the kernel (the last two dimensions).
 */
void MfCnnLayer::initOutputKernel(int paraNumFrontMaps, MfSize* paraSize)
{
    kernelSize->cloneToMe(paraSize);
    initKernel(paraNumFrontMaps);
}//Of initOutputKernel

/**
 * Initialize the bias.
 */
void MfCnnLayer::initBias()
{
    bias = new MfDoubleMatrix(1, numOutputMaps);
}//Of initBias

/**
 * Initialize the errors. The first dimension is the batch size instead of numFrontMaps.
 * paraBatchSize: the batch size.
 */
void MfCnnLayer::initErrors(int paraBatchSize)
{
    errors = new Mf4DTensor(paraBatchSize, numOutputMaps, kernelSize->width, kernelSize->height);
}//Of initErrors

