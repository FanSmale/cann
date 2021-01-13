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
#include <Matrix.h>
#include <AnnLayer.h>
#include <IntArray.h>
#include <FullAnn.h>

using namespace std;

int main()
{
    printf("Hello world!\r\n");

    //Matrix tempMatrix(2, 3);
    //tempMatrix.selfTest();

    //AnnLayer tempLayer(1, 1, 'r');
    //tempLayer.selfTest();

    //IntArray tempArray(3);
    //tempArray.selfTest();
    FullAnn tempFullAnn;
    tempFullAnn.selfTest();

    getchar();

    return 0;
}//Of main

