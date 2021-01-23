#include <math.h>
#include <stdio.h>
//#include <iostream>

#include "Activator.h"

//The default constructor
Activator::Activator()
{
    activationFunction = 's';
    activationGamma = DEFAULT_ACTIVATOR_GAMMA;
    activationBeta = DEFAULT_ACTIVATOR_BETA;
}//Of the constructor

//The second constructor
Activator::Activator(char paraActivationFunction)
{
    activationFunction = paraActivationFunction;
    activationGamma = DEFAULT_ACTIVATOR_GAMMA;
    activationBeta = DEFAULT_ACTIVATOR_BETA;
}//Of the second constructor

//Destructor
Activator::~Activator()
{
    //dtor
}//Of the destructor

//Setter
void Activator::setActivationFunction(char paraFunction)
{
    activationFunction = paraFunction;
}//Of setActivationFunction

//Setter
void Activator::setGamma(double paraGamma)
{
    activationGamma = paraGamma;
}//Of setGamma

//Setter
void Activator::setBeta(double paraBeta)
{
    activationBeta = paraBeta;
}//Of setBeta

//The sigmoid activation function
double Activator::sigmoid(double paraValue)
{
    return 1 / (1 + exp(-paraValue));
}//Of sigmoid

//The tanh activation function
double Activator::tanh(double paraValue)
{
    return (exp(paraValue) - exp(-paraValue)) / (exp(paraValue) + exp(-paraValue));
}//Of tanh

//The hard-logistic activation function
double Activator::hardLogistic(double paraValue)
{
    double tempGx = 0.25 * paraValue + 0.5;
    double resultHL = 0;
    if(tempGx >= 1)
    {
        resultHL = 1;

    }
    else if(tempGx > 0 && tempGx < 1)
    {
        resultHL = tempGx;
    }
    else
    {
        resultHL = 0;
    }//of if
    return resultHL;
}//Of hardLogistic

//The hard-tanh activation function
double Activator::hardTanh(double paraValue)
{
    double resultHT = 0;
    if(paraValue >= 1)
    {
        resultHT = 1;

    }
    else if(paraValue > -1 && paraValue < 1)
    {
        resultHT = paraValue;
    }
    else
    {
        resultHT = -1;
    }//of if
    return resultHT;
}//Of hardTanh

//The ReLU activation function
double Activator::relu(double paraValue)
{
    if(paraValue >= 0)
    {
        return paraValue;
    }
    else
    {
        return 0;
    }//of if
}//Of relu

//The LeakyReLU activation function
double Activator::leakyRelu(double paraValue, double paraGamma)
{
    if(paraValue >= 0)
    {
        return paraValue;
    }
    else
    {
        return paraGamma *  paraValue;
    }//of if
}//Of leakyRelu

//The ELU activation function
double Activator::elu(double paraValue, double paraGamma)
{
    if(paraValue >= 0)
    {
        return paraValue;
    }

    return paraGamma *  (exp(paraValue) - 1);
}//Of elu

//The Softplus activation function
double Activator::softplus(double paraValue)
{
    if(paraValue < 0)
    {
        paraValue = 0;
    }//Of if

    double tempSoft = log(1 + exp(paraValue));

    if(tempSoft >= INT_MAX)
    {
        tempSoft = INT_MAX;
    }//Of if

    return tempSoft;
}//Of softplus

//The Softsign activation function
double Activator::softsign(double paraValue)
{
    return paraValue / (1 + abs(paraValue));
}//Of softsign

//The Swish activation function
double Activator::swish(double paraValue, double paraBeta)
{
    return paraValue / (1 + exp(-paraBeta * paraValue));
}//Of swish

//The GELU activation function
double Activator::gelu(double paraValue)
{
    return paraValue / (1 + exp(-1.702 * paraValue));
}//Of gelu

//Activate
double Activator::activate(double paraValue)
{
    double resultValue = 0;
    switch (activationFunction)
    {
    case 's':
        resultValue = sigmoid(paraValue);
        break;
    case 't':
        resultValue = tanh(paraValue);
        break;
    case 'h':
        resultValue = hardLogistic(paraValue);
        break;
    case 'H':
        resultValue = hardTanh(paraValue);
        break;
    case 'r':
        resultValue = relu(paraValue);
        break;
    case 'l':
        resultValue = leakyRelu(paraValue, activationGamma);
        break;
    case 'e':
        resultValue = elu(paraValue, activationGamma);
        break;
    case 'S':
        resultValue = softplus(paraValue);
        break;
    case 'o':
        resultValue = softsign(paraValue);
        break;
    case 'w':
        resultValue = swish(paraValue, activationBeta);
        break;
    case 'g':
        resultValue = gelu(paraValue);
        break;
    default:
        break;
    }//Of switch

    //printf("paraValue = %lf, resultValue = %lf\n", paraValue, resultValue);
    return resultValue;
}//Of activate

//Code self test
void Activator::selfTest()
{
    Activator *tempActivator = new Activator('s');

    //Use functions directly
    double tempSig = tempActivator->sigmoid(5);
    printf("tempSig = %lf\n",tempSig);
    double tempTan = tempActivator->tanh(5);
    printf("tempTan = %lf\n",tempTan);
    double tempSoft = tempActivator->softplus(5);
    printf("tempSoft = %lf\n",tempSoft);

    //Now use the OOP approach
    printf("Now use the OOP approach\r\n");
    double tempValue;
    tempValue = tempActivator -> activate(5.0);
    printf("Sigmoid = %lf\r\n",tempValue);
    tempActivator->setActivationFunction('t');
    tempValue = tempActivator->activate(5.0);
    printf("Tanh = %lf\r\n",tempValue);

}//Of selfTest

