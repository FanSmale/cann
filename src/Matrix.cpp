#include <e:\c\cann\include\Matrix.h>
#include "Malloc.h"
#include <iostream>
#include <stdio.h>

using namespace std;
//The default constructor
Matrix::Matrix()
{
    data = NULL;
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
            data[i][j] = (i + 1) * (j + 1);
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
void Matrix::copy(Matrix *paraMatrix)
{
    rows = paraMatrix -> rows;
    columns = paraMatrix -> columns;
    printf("before copy data\r\n");

    //Allocate space
    data = new double *[rows];
    for (int i = 0; i < rows; i ++)
    {
        data[i] = new double[columns];
    }//Of for i

    //Copy
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = paraMatrix -> data[i][j];
        }//Of for j
    }//Of for i
}//Of copy

//Add another one with the same size
void Matrix::add(Matrix *paraMatrix)
{
    printf("Add test 1.\r\n");

    rows = paraMatrix -> rows;
    columns = paraMatrix -> columns;

    printf("Add test 2.\r\n");
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        return;
    }//Of if

    printf("Add test 3.\r\n");
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
}//Of add

//Minus another one with the same size
void Matrix::minus(Matrix *paraMatrix)
{
    rows = paraMatrix -> rows;
    columns = paraMatrix -> columns;

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
}//Of minus

//Multiply another one with the same size
void Matrix::multiply(Matrix *paraMatrix)
{
    rows = paraMatrix -> rows;
    columns = paraMatrix -> columns;

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
}//Of multiply

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

