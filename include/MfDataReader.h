/*
 * The C++ Artificial Neural network project.
 * Read the data from file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFDATAREADER_H
#define MFDATAREADER_H

#include <random>
#include <stdlib.h>
#include <string>
#include <string.h>
#include<fstream>
#include <iostream>
#include <sstream>


#include "MfIntArray.h"
#include "MfDoubleMatrix.h"

//#include "MfMath.h"

using namespace std;
using std::default_random_engine;

class MfDataReader
{
    public:
        //Empty constructor
        MfDataReader();

        //Read the data from the given file
        MfDataReader(char* paraFilename);

        //Split the data into the training and testing parts according to the given fraction
        void splitInTwo(double paraTrainingFraction);

        //Split the data according to cross-validation
        void crossValidationSplit(int paraNumFolds, int paraFoldIndex);

        //Destructor
        virtual ~MfDataReader();

        //The getter
        MfDoubleMatrix* getTrainingX();

        //The getter
        MfIntArray* getTrainingY();

        //The getter
        MfDoubleMatrix* getTestingX();

        //The getter
        MfIntArray* getTestingY();

        //The getter
        MfDoubleMatrix* getWholeX();

        //The getter
        MfIntArray* getWholeY();

        //The random array is stored in the object
        void randomize();

        //Code unit test
        void unitTest();

    protected:

        //The number of instances
        int numInstances;

        //The number of conditions
        int numConditions;

        //The number of classes
        int numClasses;

        //For data randomization
        MfIntArray* randomArray;

        //The whole input
        MfDoubleMatrix* wholeX;

        //The labels of the whole data
        MfIntArray* wholeY;

        //The training data
        MfDoubleMatrix* trainingX;

        //The labels of the training data
        MfIntArray* trainingY;

        //The testing data
        MfDoubleMatrix* testingX;

        //The labels of the testing data
        MfIntArray* testingY;

        //To generate random numbers
        default_random_engine randomEngine;

    private:
};

#endif // MFDATAREADER_H
