/*
 * The C++ Artificial Neural network project.
 * Read data from a text file, and manage splitting.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef DATAREADER_H
#define DATAREADER_H

#include <random>
#include <string>
#include<fstream>
#include <iostream>
#include <sstream>
#include <iostream>

#include "EigenSupport.h"

using std::default_random_engine;
using namespace std;

class DataReader
{
    public:
        //Empty constructor
        DataReader();

        //Read the data from the given file
        DataReader(char* paraFilename);

        //Split the data into the training and testing parts according to the given fraction
        void splitInTwo(double paraTrainingFraction);

        //Split the data according to cross-validation
        void crossValidationSplit(int paraNumFolds, int paraFoldIndex);

        //Destructor
        virtual ~DataReader();

        //The getter
        DoubleMatrix* getTrainingX();

        //The getter
        IntArray* getTrainingY();

        //The getter
        DoubleMatrix* getTestingX();

        //The getter
        IntArray* getTestingY();

        //Construct a random index array
        IntArray* getRandomIndexArray(int paraLength);

        //The random array is stored in the object
        void randomize();

        //Code unit test
        void unitTest();

    protected:

        //The number of instances
        int numInstances;

        //The number of conditions
        int numConditions;

        IntArray* randomArray;

        //The whole input
        DoubleMatrix* wholeX;

        //The labels of the whole data
        IntArray* wholeY;

        //The training data
        DoubleMatrix* trainingX;

        //The labels of the training data
        IntArray* trainingY;

        //The testing data
        DoubleMatrix* testingX;

        //The labels of the testing data
        IntArray* testingY;

        //To generate random numbers
        default_random_engine randomEngine;

    private:
};

#endif // DATAREADER_H
