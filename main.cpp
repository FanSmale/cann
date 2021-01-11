#include <iostream>
#include <stdio.h>
#include <e:\c\cann\include\Matrix.h>

using namespace std;

//Test class Matrix
int matrixTest()
{
    Matrix tempMatrix(2, 3);

    Matrix* tempTransposed = tempMatrix.transpose();
    Matrix* tempDot = tempMatrix.dot(tempTransposed);
    printf("dot\r\n");
    tempDot -> showMe();
    printf("after dot\r\n");


    Matrix tempMatrix2;
    tempMatrix2.copy(&tempMatrix);

    printf("Original\r\n");
    tempMatrix.showMe();
    tempMatrix2.showMe();

    tempMatrix.add(&tempMatrix2);
    printf("After addition\r\n");
    tempMatrix.showMe();

    tempMatrix.multiply(&tempMatrix2);
    printf("After multiply\r\n");
    tempMatrix.showMe();

    tempMatrix.minus(&tempMatrix2);
    printf("After minus\r\n");
    tempMatrix.showMe();

    return 0;
}//Of matrixTest

int main()
{
    printf("Hello world!\r\n");

    matrixTest();

    printf("Hello again!\r\n");

    getchar();

    return 0;
}

