#include <iostream>
#include <stdio.h>
#include <e:\c\cann\include\Matrix.h>
#include <e:\c\cann\include\AnnLayer.h>

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    //Matrix tempMatrix(2, 3);
    //tempMatrix.selfTest();

    AnnLayer tempLayer(1, 1, 'r');
    tempLayer.selfTest();

    getchar();

    return 0;
}

