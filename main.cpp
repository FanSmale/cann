/*
 * The C++ Artificial Neural network project.
 * The main entrance for testing.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <iostream>
#include <stdio.h>
#include "AnnLayer.h"
#include "FullAnn.h"
#include "MfMath.h"
#include "DataReader.h"

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    //Matrix tempMatrix(2, 3);
    //tempMatrix.selfTest();

    //AnnLayer tempLayer(1, 1, 'r', 0.01, 0.1);
    //tempLayer.selfTest();

    //IntArray tempArray(3);
    //tempArray.selfTest();
    //Matrix<double, 3, 3> A;

    //FullAnn tempFullAnn;
    //tempFullAnn.selfTest();

    DataReader tempReader;
    tempReader.selfTest();

    //DataReader tempReader;
    //tempReader.selfTest();

    printf("end.\r\n");
    getchar();


    return 0;
}//Of main

