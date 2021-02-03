/*
 * The C++ Artificial Neural network project.
 * Read the data from file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfDataReader.h"
#include <iostream>
using namespace std;
using std::default_random_engine;

/**
 * Empty constructor.
 */
MfDataReader::MfDataReader()
{
}//Of the default constructor

/**
 * Read the data from the given file.
 * paraFilename: the data filename.
 */
MfDataReader::MfDataReader(char* paraFilename)
{
    //Step 1. Initialize
    //printf("MfDataReader constructor test 1\r\n");
    numInstances = 0;
    numConditions = 0;
    ifstream tempInputStream(paraFilename);
    string tempLine;

    if (!tempInputStream)
    {
        throw "file not found";
    }//Of if

    double tempDouble;
    int tempInt;
    char *tempValues;
    const char * tempSplit = ",";

    //Step 2. Prepare to read data. How many instances and conditions
    //printf("MfDataReader constructor test 2\r\n");
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        if (tempLine.erase(0, tempLine.find_first_not_of(" ")) != "")
        {
            numInstances ++;
            if (numConditions == 0)
            {
                tempValues = (char *)tempLine.c_str();
                // Split the string
                char *tempRemaining = strtok(tempValues, tempSplit);
                while(tempRemaining != NULL)
                {
                    sscanf(tempRemaining, "%lf", &tempDouble);
                    tempRemaining = strtok(NULL, tempSplit);
                    numConditions ++;
                }//Of while
            }//Of if
        }//Of if
    }//Of while
    numConditions --;

    //Step 3. Allocate space
    //printf("MfDataReader constructor test 3\r\n");
    printf("Data read, numInstances = %d, numConditions = %d, the whole X is \r\n",
           numInstances, numConditions);

    wholeX = new MfDoubleMatrix(numInstances, numConditions);

    wholeY = new MfIntArray(numInstances);

    //Step 4. Now read data
    tempInputStream.clear();
    tempInputStream.seekg(0, ios::beg);

    int tempInstanceIndex = 0;
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        if (tempLine.erase(0, tempLine.find_first_not_of(" ")) != "")
        {
            stringstream tempStringStream(tempLine);
            string tempBuf;
            char *tempRemaining = (char *)tempLine.c_str();
            // Separated by comma
            tempRemaining = strtok(tempRemaining, tempSplit);
            for(int i = 0; i < numConditions; i ++)
            {
                sscanf(tempRemaining, "%lf", &tempDouble);
                //printf("MfDataReader constructor test 4.1.1, %lf\r\n", tempDouble);
                wholeX -> setValue(tempInstanceIndex, i, tempDouble);
                //wholeX[0](tempInstanceIndex, i) = tempDouble;
                tempRemaining = strtok(NULL, tempSplit);
            }//Of for i

            sscanf(tempRemaining, "%d", &tempInt);
            //printf("instance %d, y = %d.\r\n", tempInstanceIndex, tempInt);
            wholeY -> setValue(tempInstanceIndex, tempInt);
            //sscanf(tempRemaining, "%d", &[0](0, tempInstanceIndex));

            tempInstanceIndex ++;
        }//Of if
    }//Of while

    //printf("File read, wholeY is:\r\n");
    //cout << wholeY -> toString() << endl;

    tempInputStream.close();

    //Step 5. Maybe you do not want to randomize
    //printf("MfDataReader constructor test 5\r\n");
    randomArray = new MfIntArray(numInstances);
    for(int i = 0; i < numInstances; i ++)
    {
        randomArray -> setValue(i, i);
    }//Of for i

    //Step 6. Initialize other pointers
    trainingX = nullptr;
    trainingY = nullptr;
    testingX = nullptr;
    testingY = nullptr;
    //printf("MfDataReader constructor test 6\r\n");
}//Of the second constructor

/**
 * The destructor.
 */
MfDataReader::~MfDataReader()
{
    free(wholeX);
    free(wholeY);
    free(trainingX);
    free(trainingY);
    free(testingX);
    free(testingY);
    free(randomArray);
}//Of the destructor

/**
 * Split the data into the training and testing parts.
 * paraTrainingFraction: the training fraction.
 */
void MfDataReader::splitInTwo(double paraTrainingFraction)
{
    int tempTrainingSize = (int)(numInstances * paraTrainingFraction);
    int tempTestingSize = numInstances - tempTrainingSize;

    //Free space if allocated in the past
    if (trainingX != nullptr)
    {
        free(trainingX);
        free(trainingY);
        free(testingX);
        free(testingY);
    }//Of if
    trainingX = new MfDoubleMatrix(tempTrainingSize, numConditions);
    trainingY = new MfIntArray(tempTrainingSize);
    testingX = new MfDoubleMatrix(tempTestingSize, numConditions);
    testingY = new MfIntArray(tempTestingSize);

    //Training set. Support randomized order
    for(int i = 0; i < tempTrainingSize; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            //trainingX[0](i, j) = wholeX[0](randomArray[0](0, i), j);
            trainingX -> setValue(i, j, wholeX -> getValue(randomArray -> getValue(i), j));
        }//Of for j
        //trainingY[0](0, i) = wholeY[0](0, randomArray[0](0, i));
        trainingY -> setValue(i, wholeY -> getValue(randomArray -> getValue(i)));
    }//Of for i
    //printf("In MfDataReader, the training Y is:\r\n");
    //cout << trainingY -> toString() << endl;

    //Testing set
    for(int i = tempTrainingSize; i < numInstances; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            //testingX[0](i - tempTrainingSize, j) = wholeX[0](randomArray[0](0, i), j);
            testingX -> setValue(i - tempTrainingSize, j, wholeX -> getValue(randomArray -> getValue(i), j));
        }//Of for j
        //testingY[0](0, i - tempTrainingSize) = wholeY[0](0, randomArray[0](0, i));
        testingY -> setValue(i - tempTrainingSize, wholeY -> getValue(randomArray -> getValue(i)));
    }//Of for i
}//Of splitInTwo

/**
 * Split the data according to cross-validation.
 *   Only one fold is generated at a time.
 *   For full CV, please invoke this method in a loop with
 *   paraFoldIndex ranging [0 .. paraNumFolds-1].
 * paraNumFolds: the number of folds.
 * paraFoldIndex: the index of the current fold.
 */
void MfDataReader::crossValidationSplit(int paraNumFolds, int paraFoldIndex)
{
    int tempTestingSize = numInstances / paraNumFolds;
    if (paraFoldIndex < numInstances % paraNumFolds)
    {
        tempTestingSize ++;
    }//Of if
    int tempTrainingSize = numInstances - tempTestingSize;
    printf("The training size is %d\r\n", tempTrainingSize);

    //Free space if allocated in the past
    if (trainingX != nullptr)
    {
        free(trainingX);
        free(trainingY);
        free(testingX);
        free(testingY);
    }//Of if
    trainingX = new MfDoubleMatrix(tempTrainingSize, numConditions);
    trainingY = new MfIntArray(tempTrainingSize);
    testingX = new MfDoubleMatrix(tempTestingSize, numConditions);
    testingY = new MfIntArray(tempTestingSize);
    int tempTrainingIndex = 0;
    int tempTestingIndex = 0;

    for(int i = 0; i < numInstances; i ++)
    {
        if (i % paraNumFolds != paraFoldIndex)
        {
            for(int j = 0; j < numConditions; j ++)
            {
                //trainingX[0](tempTrainingIndex, j) = wholeX[0](randomArray[0](0, i), j);
                trainingX -> setValue(tempTrainingIndex, j, wholeX -> getValue(randomArray -> getValue(i), j));
            }//Of for j
            //trainingY[0](0, tempTrainingIndex) = wholeY[0](0, randomArray[0](0, i));
            trainingY -> setValue(tempTrainingIndex, wholeY -> getValue(randomArray -> getValue(i)));
            tempTrainingIndex ++;
        }
        else
        {
            for(int j = 0; j < numConditions; j ++)
            {
                //testingX[0](tempTestingIndex, j) = wholeX[0](randomArray[0](0, i), j);
                testingX -> setValue(tempTestingIndex, j, wholeX -> getValue(randomArray -> getValue(i), j));
            }//Of for j
            //testingY[0](0, tempTestingIndex) = wholeY[0](0, randomArray[0](0, i));
            testingY -> setValue(tempTestingIndex, wholeY -> getValue(randomArray -> getValue(i)));
            tempTestingIndex ++;
        }//Of if
    }//Of for i
}//Of crossValidationSplit

/**
 * Getter.
 */
MfDoubleMatrix* MfDataReader::getTrainingX()
{
    return trainingX;
}//Of getTrainingX

/**
 * Getter.
 */
MfIntArray* MfDataReader::getTrainingY()
{
    return trainingY;
}//Of getTrainingY

/**
 * Getter.
 */
MfDoubleMatrix* MfDataReader::getTestingX()
{
    return testingX;
}//Of getTestingX

/**
 * Getter.
 */
MfIntArray* MfDataReader::getTestingY()
{
    return testingY;
}//Of getTestingY

/**
 * Randomize the data through generating a random int array.
 *   Data are accessed through indirect addressing.
 */
void MfDataReader::randomize()
{
    randomArray -> randomizeOrder();
}//Of randomize

/**
 * Code unit test.
 */
void MfDataReader::unitTest()
{
    string tempString = "d:\\c\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();
    //char *s_input = (char *)tempString.c_str();

    MfDataReader tempReader(tempFilename);

    //printf("Before ranomize \r\n");
    tempReader.randomize();
    //printf("After ranomize \r\n");
    tempReader.splitInTwo(0.6);

    printf("The training X is: \r\n");
    cout << tempReader.trainingX -> toString() << endl;
    //printf("The training X is: \r\n");

    //cout << tempArrayPtr[0] << endl;
}//Of unitTest
