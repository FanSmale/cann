#include <iostream>
#include <stdio.h>
#include <e:\c\cann\include\Matrix.h>

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    Matrix tempMatrix(2, 3);
    tempMatrix.selfTest();

    getchar();

    return 0;
}

