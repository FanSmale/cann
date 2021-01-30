#include<fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iostream>

#include "DataReader.h"
#include "MfMath.h"

using namespace std;

/**
 * Empty constructor.
 */
DataReader::DataReader()
{
}//Of the default constructor

/**
 * Read the data from the given file.
 * paraFilename: the data filename.
 */
DataReader::DataReader(char* paraFilename)
{
    //Step 1. Initialize
    numInstances = 0;
    numConditions = 0;
    ifstream tempInputStream(paraFilename);
    string tempLine;

    if (!tempInputStream)
    {
        throw FILE_NOT_EXISTS_EXCEPTION;
    }//Of if

    double tempDouble;
    char *tempValues;
    const char * tempSplit = ",";

    //Step 2. Prepare to read data. How many instances and conditions
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

    //Step 3. Allocate space
    numConditions --;
    wholeX = new DoubleMatrix(numInstances, numConditions);
    wholeY = new IntArray(1, numInstances);

    printf("Data read, numInstances = %d, numConditions = %d, the whole X is \r\n",
           numInstances, numConditions);

    //Step 4. Now read data
    tempInputStream.close();
    tempInputStream.open(paraFilename);
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
                sscanf(tempRemaining, "%lf", &wholeX[0](tempInstanceIndex, i));
                tempRemaining = strtok(NULL, tempSplit);
            }//Of for i

            sscanf(tempRemaining, "%d", &wholeY[0](0, tempInstanceIndex));
            tempInstanceIndex ++;
        }//Of if
    }//Of while

    tempInputStream.close();

    //Step 5. Maybe you do not want to randomize
    randomArray = new IntArray(1, numInstances);
    for(int i = 0; i < numInstances; i ++)
    {
        randomArray[0](0, i) = i;
    }//Of for i

    //Step 6. Initialize other pointers
    trainingX = nullptr;
    trainingY = nullptr;
    testingX = nullptr;
    testingY = nullptr;
}//Of the second constructor

/**
 * The destructor.
 */
DataReader::~DataReader()
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
void DataReader::splitInTwo(double paraTrainingFraction)
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
    trainingX = new DoubleMatrix(tempTrainingSize, numConditions);
    trainingY = new IntArray(1, tempTrainingSize);
    testingX = new DoubleMatrix(tempTestingSize, numConditions);
    testingY = new IntArray(1, tempTestingSize);

    //Training set. Support randomized order
    for(int i = 0; i < tempTrainingSize; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            trainingX[0](i, j) = wholeX[0](randomArray[0](0, i), j);
        }//Of for j
        trainingY[0](0, i) = wholeY[0](0, randomArray[0](0, i));
    }//Of for i

    //Testing set
    for(int i = tempTrainingSize; i < numInstances; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            testingX[0](i - tempTrainingSize, j) = wholeX[0](randomArray[0](0, i), j);
        }//Of for j
        testingY[0](0, i - tempTrainingSize) = wholeY[0](0, randomArray[0](0, i));
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
void DataReader::crossValidationSplit(int paraNumFolds, int paraFoldIndex)
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
    trainingX = new DoubleMatrix(tempTrainingSize, numConditions);
    trainingY = new IntArray(1, tempTrainingSize);
    testingX = new DoubleMatrix(tempTestingSize, numConditions);
    testingY = new IntArray(1, tempTestingSize);
    int tempTrainingIndex = 0;
    int tempTestingIndex = 0;

    for(int i = 0; i < numInstances; i ++)
    {
        if (i % paraNumFolds != paraFoldIndex)
        {
            for(int j = 0; j < numConditions; j ++)
            {
                trainingX[0](tempTrainingIndex, j) = wholeX[0](randomArray[0](0, i), j);
            }//Of for j
            trainingY[0](0, tempTrainingIndex) = wholeY[0](0, randomArray[0](0, i));
            tempTrainingIndex ++;
        }
        else
        {
            for(int j = 0; j < numConditions; j ++)
            {
                testingX[0](tempTestingIndex, j) = wholeX[0](randomArray[0](0, i), j);
            }//Of for j
            testingY[0](0, tempTestingIndex) = wholeY[0](0, randomArray[0](0, i));
            tempTestingIndex ++;
        }//Of if
    }//Of for i
}//Of crossValidationSplit

/**
 * Getter.
 */
DoubleMatrix* DataReader::getTrainingX()
{
    return trainingX;
}//Of getTrainingX

/**
 * Getter.
 */
IntArray* DataReader::getTrainingY()
{
    return trainingY;
}//Of getTrainingY

/**
 * Getter.
 */
DoubleMatrix* DataReader::getTestingX()
{
    return testingX;
}//Of getTestingX

/**
 * Getter.
 */
IntArray* DataReader::getTestingY()
{
    return testingY;
}//Of getTestingY

/**
 * Construct an index array as a randomization of [0, paraLength - 1].
 * paraLength: the length of the array.
 */
IntArray* DataReader::getRandomIndexArray(int paraLength)
{
    IntArray* tempArrayPtr = new IntArray(1, paraLength);
    int tempFirstIndex, tempSecondIndex;
    for(int i = 0; i < paraLength; i++)
    {
        tempArrayPtr[0](0, i) = i;
    }//Of for i

    int tempValue;
    for(int i = 0; i < paraLength * 10; i++)
    {
        tempFirstIndex = randomEngine() % paraLength;
        tempSecondIndex = randomEngine() % paraLength;
        tempValue = tempArrayPtr[0](0, tempFirstIndex);
        tempArrayPtr[0](0, tempFirstIndex) = tempArrayPtr[0](0, tempSecondIndex);
        tempArrayPtr[0](0, tempSecondIndex) = tempValue;
    }//Of for i

    return tempArrayPtr;
}//Of getRandomIndexArray

/**
 * Randomize the data through generating a random int array.
 *   Data are accessed through indirect addressing.
 */
void DataReader::randomize()
{
    free(randomArray);
    randomArray = getRandomIndexArray(numInstances);
}//Of randomize

/**
 * Code unit test.
 */
void DataReader::unitTest()
{
    string tempString = "d:\\c\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();
    //char *s_input = (char *)tempString.c_str();

    DataReader tempReader(tempFilename);

    tempReader.randomize();
    tempReader.splitInTwo(0.6);

    printf("The training X is: \r\n");
    cout << tempReader.wholeX[0] << endl;
    //printf("The training X is: \r\n");

    //cout << tempArrayPtr[0] << endl;
}//Of unitTest
