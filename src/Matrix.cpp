#include <e:\c\cann\include\Matrix.h>
#include "Malloc.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

//The default constructor
Matrix::Matrix()
{
    data = nullptr;
}

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
            //data[i][j] = (i + 1) * (j + 1);
        }//Of for j
    }//Of for i
}

//Destructor
Matrix::~Matrix()
{
    free(data);
    //delete data;
}

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
        return nullptr;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return nullptr;
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
        printf("Rows do not match.");
        return;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return;
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
        printf("Rows do not match.");
        return nullptr;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return nullptr;
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
        return;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return;
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
        return nullptr;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return nullptr;
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
        return;
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        return;
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
        return nullptr;
    }//Of if

    Matrix* newMatrixPtr = new Matrix(tempRows, tempColumns);
    double tempValue = 0;

    for (int i = 0; i < tempRows; i ++)
    {
        for (int j = 0; j < tempColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < columns; k ++) {
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

//Show me
void Matrix::showMe()
{
    printf("I am a matrix with size %d * %d. \r\n", rows, columns);
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            printf("%f, ", data[i][j]);
        }//Of for j
        printf("\r\n");
    }//Of for i

    printf("matrix ends. \r\n");

    return;
}//Of showMe

//Code self test
void Matrix::selfTest()
{
    Matrix* tempMatrix = new Matrix(2, 3);
    printf("Original\r\n");
    tempMatrix -> showMe();

    Matrix* tempMatrix2 = tempMatrix -> copy();
    printf("Copy\r\n");
    tempMatrix2 -> showMe();

    Matrix* tempTransposed = tempMatrix -> transpose();
    printf("Transpose\r\n");
    tempTransposed -> showMe();

    Matrix* tempDot = tempMatrix -> dot(tempTransposed);
    printf("Dot\r\n");
    tempDot -> showMe();

    Matrix* tempAdded = tempMatrix -> add(tempMatrix2);
    printf("Add\r\n");
    tempAdded -> showMe();

    Matrix* tempMultiply = tempMatrix -> multiply(tempMatrix2);
    printf("Multiply\r\n");
    tempMultiply -> showMe();

    Matrix* tempMinus = tempMultiply  -> minus(tempMatrix);
    printf("Minus\r\n");
    tempMinus -> showMe();
}//Of selfTest
