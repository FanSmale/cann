#include <iostream>
#include <stdio.h>
#include <e:\c\cann\include\Matrix.h>

using namespace std;

//Test class Matrix
int matrixTest()
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

    return 0;
}//Of matrixTest

int main()
{
    printf("Hello world!\r\n");

    Matrix tempMatrix(2, 3);
    tempMatrix.selfTest();

    getchar();

    return 0;
}

