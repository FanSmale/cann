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
#include "EigenSupport.h"
#include "DataReader.h"
#include "CnnLayer.h"
#include "MfIntArray.h"
#include "MfDoubleMatrix.h"
#include "MfDataReader.h"
#include "MfAnnLayer.h"
#include "MfFullAnn.h"

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

    //DataReader tempReader;
    //tempReader.selfTest();

    //DataReader tempReader;
    //tempReader.selfTest();

    //Activator tempActivator;
    //tempActivator.selfTest();

    //FullAnn tempFullAnn;
    //tempFullAnn.trainingTestingTest();
    //tempFullAnn.crossValidationTest();

    //CnnLayer tempCnnLayer;
    //tempCnnLayer.unitTest();

    //MfIntArray tempArray;
    //tempArray.unitTest();

    //MfDoubleMatrix tempMatrix;
    //tempMatrix.unitTest();

    //MfDataReader tempReader;
    //tempReader.unitTest();

    //MfAnnLayer tempLayer;
    //tempLayer.unitTest();

    MfFullAnn tempMfFullAnn;
    //tempMfFullAnn.trainingTestingTest();
    tempMfFullAnn.crossValidationTest();

    printf("end.\r\n");
    getchar();

    return 0;
}//Of main

