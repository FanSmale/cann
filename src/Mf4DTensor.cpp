/*
 * The C++ Artificial Neural network project.
 * 4D tensor for CNN.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */
#include "Mf4DTensor.h"

/**
 * The default constructor.
 */
Mf4DTensor::Mf4DTensor()
{
    firstLength = 0;
    secondLength = 0;
    thirdLength = 0;
    fourthLength = 0;
    data = nullptr;
}//Of the default constructor

/**
 * The second constructor. All values are initialized as a random value in [0, 1].
 */
Mf4DTensor::Mf4DTensor(int paraFirstLength, int paraSecondLength, int paraThirdLength, int paraFourthLength)
{
    firstLength = paraFirstLength;
    secondLength = paraSecondLength;
    thirdLength = paraThirdLength;
    fourthLength = paraFourthLength;

    data = new double***[firstLength];
    for(int i = 0; i < firstLength; i ++)
    {
        data[i] = new double**[secondLength];
        for(int j = 0; j < secondLength; j ++)
        {
            data[i][j] = new double*[thirdLength];
            for(int k = 0; k < thirdLength; k ++)
            {
                data[i][j][k] = new double[fourthLength];
                for(int i1 = 0; i1 < fourthLength; i1 ++)
                {
                    data[i][j][k][i1] = random0To1();
                }//Of for i1
            }//Of for k
        }//Of for j
    }//Of for i
}//Of the second constructor

/**
 * The destructor.
 */
Mf4DTensor::~Mf4DTensor()
{
    for(int i = 0; i < firstLength; i ++)
    {
        for(int j = 0; j < secondLength; j ++)
        {
            for(int k = 0; k < thirdLength; k ++)
            {
                free(data[i][j][k]);
            }//Of for k
            free(data[i][j]);
        }//Of for j
        free(data[i]);
    }//Of for i

    free(data);
}//Of the destructor

/**
 * Fill with the given value.
 * paraValue: the given value.
 */
void Mf4DTensor::fill(double paraValue)
{
    for(int i = 0; i < firstLength; i ++)
    {
        for(int j = 0; j < secondLength; j ++)
        {
            for(int k = 0; k < thirdLength; k ++)
            {
                for(int i1 = 0; i1< fourthLength; i1 ++)
                {
                    data[i][j][k][i1] = paraValue;
                }//Of for i1
            }//Of for k
        }//Of for j
    }//Of for i
}//Of fill


/**
 * Fill with random values.
 * paraLowerBound: the lower bound.
 * paraUpperBound: the upper bound
 */
void Mf4DTensor::fill(double paraLowerBound, double paraUpperBound)
{
    for(int i = 0; i < firstLength; i ++)
    {
        for(int j = 0; j < secondLength; j ++)
        {
            for(int k = 0; k < thirdLength; k ++)
            {
                for(int i1 = 0; i1< fourthLength; i1 ++)
                {
                    data[i][j][k][i1] = rand() * (paraUpperBound - paraLowerBound) /RAND_MAX + paraLowerBound;
                }//Of for i1
            }//Of for k
        }//Of for j
    }//Of for i
}//Of fill

/**
 * Convert to string for display.
 * Returns: The string showing myself.
 */
string Mf4DTensor::toString()
{
    string resultString = "[\r\n";
    for(int i = 0; i < firstLength; i ++)
    {
        resultString += "[\r\n";
        for(int j = 0; j < secondLength; j ++)
        {
            resultString += "[";
            for(int k = 0; k < thirdLength; k ++)
            {
                resultString += "[";
                for(int i1 = 0; i1< fourthLength; i1 ++)
                {
                    resultString += to_string(data[i][j][k][i1]) + ", ";
                }//Of for i1
                resultString += "]";
            }//Of for k
            resultString += "]\r\n";
        }//Of for j
    resultString += "]\r\n";
    }//Of for i
    resultString += "]\r\n";

    return resultString;
}//Of toString

/**
 * Set one value in the tensor.
 * paraValue: the given value.
 */
void Mf4DTensor::setValue(int paraFirstIndex, int paraSecondIndex, int paraThirdIndex, int paraFourthIndex, double paraValue)
{
    data[paraFirstIndex][paraSecondIndex][paraThirdIndex][paraFourthIndex] = paraValue;
}//Of setValue

/**
 * Sum to one matrix.
 * paraIndex: an index for the second dimension.
 * paraMatrix: the matrix.
 */
void Mf4DTensor::sumToMatrix(int paraIndex, MfDoubleMatrix* paraMatrix)
{
    int tempRows = paraMatrix->getRows();
    int tempColumns = paraMatrix->getColumns();
    double tempSum = 0;
    for (int mi = 0; mi < tempRows; mi++)
    {
        for (int nj = 0; nj < tempColumns; nj++)
        {
            tempSum = 0;
            for (int i = 0; i < firstLength; i++)
            {
                tempSum += data[i][paraIndex][mi][nj];
            }//Of for i
            paraMatrix->setValue(mi, nj, tempSum);
        }//Of for nj
    }//Of for mi
}//Of sumToMatrix

/**
 * Code unit test.
 */
void Mf4DTensor::unitTest()
{
    Mf4DTensor* tempTensor = new Mf4DTensor(2, 2, 2, 2);
    printf(tempTensor->toString().data());
}//Of unitTest
