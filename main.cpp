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

//For ANN
#include "MfIntArray.h"
#include "MfDoubleMatrix.h"
#include "MfDataReader.h"
#include "MfAnnLayer.h"
#include "MfFullAnn.h"

//For CNN
#include "MfSize.h"
#include "Mf4DTensor.h"
#include "MfFullCnn.h"

//For RNN
#include "MfDoubleMatrixArray.h"
#include "MfRnnLayer.h"

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    /**
     *For basis
     */
    //MfIntArray tempArray;
    //tempArray.unitTest();

    //MfDoubleMatrix tempMatrix;
    //tempMatrix.unitTest();

    //MfDataReader tempReader;
    //tempReader.unitTest();

    /**
     *For Ann
     */
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
    //tempCnn.integratedTest();
    tempCnn.mnistTest();

   /**
    * For Rnn
    */
    //MfDoubleMatrixArray* tempArray = new MfDoubleMatrixArray();
    //tempArray->unitTest();

    printf("end.\r\n");
    getchar();

    return 0;
}//Of main

