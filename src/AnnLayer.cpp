#include "AnnLayer.h"
#include "Malloc.h"
#include "Math.h"
#include "stdio.h"

//The default constructor
AnnLayer::AnnLayer()
{
    inputSize = 1;
    outputSize = 1;
}//Of the default constructor

//Constructor for input/output size
AnnLayer::AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation)
{
    inputSize = paraInputSize;
    outputSize = paraOutputSize;

    weightMatrix = new Matrix(paraInputSize, paraOutputSize);
    offsetMatrix = new Matrix(1, paraOutputSize);

    activation = paraActivation;
}//Of the second constructor

//Destructor
AnnLayer::~AnnLayer()
{
    free(weightMatrix);
}//Of the destructor

//Activate
Matrix* AnnLayer::forward(Matrix* paraData)
{
    printf("Forwarding, the data is: \r\n");
    paraData -> showMe();

    printf("The weights are: \r\n");
    weightMatrix -> showMe();

    Matrix* resultData = paraData -> dot(weightMatrix);
    resultData -> addToMe(offsetMatrix);
    resultData -> activate(activation);

    printf("The resultData are: \r\n");
    resultData -> showMe();
    return resultData;
}//Of forward

//Code self test
void AnnLayer::selfTest()
{
    AnnLayer* tempLayer = new AnnLayer(2, 3, 's');
    Matrix* tempInput = new Matrix(1, 2);
    printf("The input is: \r\n");
    tempInput -> showMe();

    printf("The weights are: \r\n");
    tempLayer -> weightMatrix -> showMe();

    Matrix* tempOutput = tempLayer -> forward(tempInput);
    printf("The output is: \r\n");
    tempOutput -> showMe();
}//Of selfTest
