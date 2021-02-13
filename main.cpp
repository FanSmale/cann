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
//#include "AnnLayer.h"
//#include "FullAnn.h"
//#include "EigenSupport.h"
//#include "DataReader.h"
//#include "CnnLayer.h"
#include "MfIntArray.h"
#include "MfDoubleMatrix.h"
#include "MfDataReader.h"
#include "MfAnnLayer.h"
#include "MfFullAnn.h"

#include "MfSize.h"
#include "Mf4DTensor.h"
#include "MfFullCnn.h"

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    /**
     *For Ann
     */
    //MfIntArray tempArray;
    //tempArray.unitTest();

    //MfDoubleMatrix tempMatrix;
    //tempMatrix.unitTest();

    //MfDataReader tempReader;
    //tempReader.unitTest();

    //MfAnnLayer tempLayer;
    //tempLayer.unitTest();

    //MfFullAnn tempMfFullAnn;
    //tempMfFullAnn.trainingTestingTest();
    //tempMfFullAnn.crossValidationTest();

    /**
     *For Cnn
     */
    //MfSize tempSize;
    //tempSize.unitTest();

    //Mf4DTensor* tempTensor = new Mf4DTensor();
    //tempTensor->unitTest();

    MfFullCnn tempCnn;
    tempCnn.trainingTestingTest();
    //tempMfFullAnn.crossValidationTest();

    printf("end.\r\n");
    getchar();

    return 0;
}//Of main

