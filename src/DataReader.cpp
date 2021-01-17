#include <iostream>
#include "DataReader.h"

using namespace std;

//Empty constructor
DataReader::DataReader()
{
    //ctor
}//Of the default constructor

//Read the data from the given file
DataReader::DataReader(char* paraFilename)
{
    //ctor
    int tempNumInstances;
    int tempNumConditions;
}//Of the second constructor

DataReader::~DataReader()
{
    //free();

}//Of the destructor

//Split the data into the training and testing parts according to the given fraction
void DataReader::splitInTwo(double paraTrainingFraction)
{

}//Of splitInTwo

//The getter
DoubleMatrix* DataReader::getTrainingX()
{
    return &trainingX;
}//Of getTrainingX

//The getter
IntArray* DataReader::getTrainingY()
{
    return &trainingY;
}//Of getTrainingY

//The getter
DoubleMatrix* DataReader::getTestingX()
{
    return &testingX;
}//Of getTestingX

//The getter
IntArray* DataReader::getTestingY()
{
    return &testingY;
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

//Code self test
void DataReader::selfTest()
{
    DataReader tempReader;
    IntArray* tempArrayPtr = tempReader.getRandomIndexArray(10);

    cout << tempArrayPtr[0] << endl;
}//Of selfTest
