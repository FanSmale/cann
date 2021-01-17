<<<<<<< HEAD
#include<fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "DataReader.h"
#include "MfMath.h"
=======
#include <iostream>
#include "DataReader.h"
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001

using namespace std;

//Empty constructor
DataReader::DataReader()
{
    //ctor
}//Of the default constructor

//Read the data from the given file
DataReader::DataReader(char* paraFilename)
{
<<<<<<< HEAD
    //Initialize
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
    //How many instances and conditions
    while (getline(tempInputStream, tempLine)) // line�в�����ÿ�еĻ��з�
    {
        if (tempLine.erase(0, tempLine.find_first_not_of(" ")) != "")
        {
            numInstances ++;
            if (numConditions == 0)
            {
                tempValues = (char *)tempLine.c_str();
                // �Զ���Ϊ�ָ�������ַ���
                char *tempRemaining = strtok(tempValues, tempSplit);
                while(tempRemaining != NULL)
                {
                    // char * -> int
                    sscanf(tempRemaining, "%lf", &tempDouble);
                    tempRemaining = strtok(NULL, tempSplit);
                    numConditions ++;
                }//Of while
            }//Of if
        }//Of if
    }//Of while

    //Allocate space
    numConditions --;
    wholeX = new DoubleMatrix(numInstances, numConditions);
    wholeY = new IntArray(1, numInstances);

    printf("Data read, numInstances = %d, numConditions = %d, the whole X is \r\n",
        numInstances, numConditions);
    //DoubleMatrix tempMatrix = wholeX[0];
    //cout << tempMatrix(0, 0) << endl;

    //Now read data
    tempInputStream.close();
    tempInputStream.open(paraFilename);
    int tempInstanceIndex = 0;
    //int tempConditionIndex = 0;
    while (getline(tempInputStream, tempLine)) // line�в�����ÿ�еĻ��з�
    {
        //tempConditionIndex = 0;
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
            //printf("%d\r\n", wholeY[0](0, tempInstanceIndex));
            tempInstanceIndex ++;
        }//Of if
    }//Of while

    /*
    printf("The pointer is: ");
    cout << wholeX << endl;
    DoubleMatrix tempMatrix = wholeX[0];
    cout << tempMatrix << endl;
    cout << tempMatrix(0, 0) << endl;
    printf("The first value shown. ");
    */

    tempInputStream.close();
=======
    //ctor
    int tempNumInstances;
    int tempNumConditions;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of the second constructor

DataReader::~DataReader()
{
<<<<<<< HEAD
    free(wholeX);
    free(wholeY);
    free(trainingX);
    free(trainingY);
    free(testingX);
    free(testingY);
=======
    //free();

>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of the destructor

//Split the data into the training and testing parts according to the given fraction
void DataReader::splitInTwo(double paraTrainingFraction)
{
<<<<<<< HEAD
    int tempTrainingSize = (int)(numInstances * paraTrainingFraction);
    int tempTestingSize = numInstances - tempTrainingSize;

    trainingX = new DoubleMatrix(tempTrainingSize, numConditions);
    trainingY = new IntArray(1, tempTrainingSize);
    testingX = new DoubleMatrix(tempTestingSize, numConditions);
    testingY = new IntArray(1, tempTestingSize);

    //randomArray[0](0, i) support randomized order
    for(int i = 0; i < tempTrainingSize; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            trainingX[0](i, j) = wholeX[0](randomArray[0](0, i), j);
        }//Of for j
        trainingY[0](0, i) = wholeY[0](0, randomArray[0](0, i));
    }//Of for i

    for(int i = tempTrainingSize; i < numInstances; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            testingX[0](i - tempTrainingSize, j) = wholeX[0](randomArray[0](0, i), j);
        }//Of for j
        testingY[0](0, i - tempTrainingSize) = wholeY[0](0, randomArray[0](0, i));
    }//Of for i
=======

>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of splitInTwo

//The getter
DoubleMatrix* DataReader::getTrainingX()
{
<<<<<<< HEAD
    return trainingX;
=======
    return &trainingX;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of getTrainingX

//The getter
IntArray* DataReader::getTrainingY()
{
<<<<<<< HEAD
    return trainingY;
=======
    return &trainingY;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of getTrainingY

//The getter
DoubleMatrix* DataReader::getTestingX()
{
<<<<<<< HEAD
    return testingX;
=======
    return &testingX;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of getTestingX

//The getter
IntArray* DataReader::getTestingY()
{
<<<<<<< HEAD
    return testingY;
=======
    return &testingY;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of getTestingY

//Construct a random index array
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

<<<<<<< HEAD

//Construct a random index array
void DataReader::randomize()
{
    randomArray = getRandomIndexArray(numInstances);
}//Of randomize

//Code self test
void DataReader::selfTest()
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
=======
//Code self test
void DataReader::selfTest()
{
    DataReader tempReader;
    IntArray* tempArrayPtr = tempReader.getRandomIndexArray(10);

    cout << tempArrayPtr[0] << endl;
>>>>>>> 56e6ec17f478bfda9b0d79871de29d1189a18001
}//Of selfTest
