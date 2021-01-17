#ifndef DATAREADER_H
#define DATAREADER_H

#include <random>
<<<<<<< HEAD
#include <string>

=======
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
#include "MfMath.h"

using std::default_random_engine;

class DataReader
{
    public:
        //Empty constructor
        DataReader();

        //Read the data from the given file
        DataReader(char* paraFilename);

        //Split the data into the training and testing parts according to the given fraction
        void splitInTwo(double paraTrainingFraction);

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

<<<<<<< HEAD
        //The random array is stored in the object
        void randomize();

=======
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
        //Code self test
        void selfTest();

    protected:

<<<<<<< HEAD
        //The number of instances
        int numInstances;

        //The number of conditions
        int numConditions;

        IntArray* randomArray;

=======
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
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
