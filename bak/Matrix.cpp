/*
 * The C++ Artificial Neural network project.
 * This class manages matrices.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "Malloc.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Matrix.h>
#include <MfMath.h>
#include <string>

using namespace std;

//The default constructor
Matrix::Matrix()
{
    rows = 0;
    columns = 0;
    data = nullptr;
}//Of the default constructor

//Initialize a matrix with given sizes
Matrix::Matrix(int paraRows, int paraColumns)
{
    rows = paraRows;
    columns = paraColumns;

    //Allocate space
    data = new double *[rows];
    for (int i = 0; i < rows; i ++)
    {
        data[i] = new double[columns];
    }//Of for i

    //Some initial values
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = rand() / (double)(RAND_MAX);
        }//Of for j
    }//Of for i
}//Of the second constructor

//Destructor
Matrix::~Matrix()
{
    free(data);
}//Of the destructor

//Show me with a string for display
string Matrix::toString()
{
    string resultString = "I am a matrix with size " + to_string(rows)
        + "*" + to_string(columns) + "\r\n";
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultString += to_string(data[i][j]) +  ", ";
        }//Of for j
        resultString += "\r\n";
    }//Of for i

    resultString += "matrix ends. \r\n";

    return resultString;
}//Of toString

//Get a value at the given position
double Matrix::getValue(int paraRow, int paraColumn)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("Matrix.getValue() out of range.");
        throw OUT_OF_RANGE_EXCEPTION;
    }//Of if
    return data[paraRow][paraColumn];
}//Of getValue

//Set a value at the given position
double Matrix::setValue(int paraRow, int paraColumn, double paraValue)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("Matrix.setValue() out of range.");
        throw OUT_OF_RANGE_EXCEPTION;
    }//Of if
    data[paraRow][paraColumn] = paraValue;
}//Of setValue

//Copy a matrix
Matrix* Matrix::copy()
{
    Matrix* resultMatrix = new Matrix(rows, columns);

    //Copy
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultMatrix -> data[i][j] = data[i][j];
        }//Of for j
    }//Of for i

    return resultMatrix;
}//Of copy

//Add another one with the same size
Matrix* Matrix::add(Matrix* paraMatrix)
{
    Matrix* resultMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultMatrix -> data[i][j] += paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i

    return resultMatrix;
}//Of add

//Add another one with the same size to me
void Matrix::addToMe(Matrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Matrix.addToMe(), rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Matrix.addToMe(), columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] += paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i
}//Of addToMe

//Minus another one with the same size
Matrix* Matrix::minus(Matrix* paraMatrix)
{
    Matrix* resultMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("Matrix.minus(), rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Matrix.minus(), columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultMatrix -> data[i][j] -= paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i

    return resultMatrix;
}//Of minus

//Minus another one with the same size to me
void Matrix::minusToMe(Matrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] -= paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i
}//Of minusToMe

//Multiply another one with the same size
Matrix* Matrix::multiply(Matrix* paraMatrix)
{
    Matrix* resultMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultMatrix -> data[i][j] *= paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i

    return resultMatrix;
}//Of multiply

//Multiply another one with the same size to me
void Matrix::multiplyToMe(Matrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] *= paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i
}//Of multiplyToMe

//Dot multiply, return a new matrix
Matrix* Matrix::dot(Matrix *paraMatrix)
{
    int tempRows = rows;
    int tempColumns = paraMatrix -> columns;

    if (columns != paraMatrix -> rows)
    {
        printf("Matrices do not match.");
        throw SIZE_NOT_MATCH_EXCEPTION;
    }//Of if

    Matrix* newMatrixPtr = new Matrix(tempRows, tempColumns);
    double tempValue = 0;

    for (int i = 0; i < tempRows; i ++)
    {
        for (int j = 0; j < tempColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < columns; k ++)
            {
                tempValue += data[i][k] * paraMatrix -> data[k][j];
            }//Of for k
            newMatrixPtr -> data[i][j] = tempValue;
        }//Of for j
    }//Of for i

    return newMatrixPtr;
}//Of dot

//Transpose, return a new matrix
Matrix* Matrix::transpose()
{
    int tempRows = columns;
    int tempColumns = rows;

    Matrix* newMatrixPtr = new Matrix(tempRows, tempColumns);

    for (int i = 0; i < tempRows; i ++)
    {
        for (int j = 0; j < tempColumns; j ++)
        {
            newMatrixPtr -> data[i][j] = data[j][i];
        }//Of for j
    }//Of for i

    return newMatrixPtr;
}//Of transpose

//Activate myself
void Matrix::activate(char paraFunction)
{
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = activate(data[i][j], paraFunction);
        }//Of for j
    }//Of for i
}//Of activate

//Activate for the given value, independent of this object
double Matrix::activate(double paraValue, char paraFunction)
{
    switch (paraFunction)
    {
    case 's':
        return 1 / (1 + exp(-paraValue));
    case 'r':
        if (paraValue > 0)
        {
            return paraValue;
        }
        else
        {
            return 0;
        }//Of if
    default:
        return paraValue;
    }//Of switch
}//Of activate

//Code self test
void Matrix::selfTest()
{
    Matrix* tempMatrix = new Matrix(2, 3);
    printf("Original\r\n");

    printf(tempMatrix -> toString().data());

    Matrix* tempMatrix2 = tempMatrix -> copy();
    printf("Copy\r\n");
    printf(tempMatrix2 -> toString().data());

    Matrix* tempTransposed = tempMatrix -> transpose();
    printf("Transpose\r\n");
    printf(tempTransposed -> toString().data());

    Matrix* tempDot = tempMatrix -> dot(tempTransposed);
    printf("Dot\r\n");
    printf(tempDot -> toString().data());

    Matrix* tempAdded = tempMatrix -> add(tempMatrix2);
    printf("Add\r\n");
    printf(tempAdded -> toString().data());

    Matrix* tempMultiply = tempMatrix -> multiply(tempMatrix2);
    printf("Multiply\r\n");
    printf(tempMultiply -> toString().data());

    Matrix* tempMinus = tempMultiply  -> minus(tempMatrix);
    printf("Minus\r\n");
    printf(tempMinus -> toString().data());
}//Of selfTest
