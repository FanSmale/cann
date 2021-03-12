/*
 * The C++ Artificial Neural network project.
 * Read the data from file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <iostream>

#include "MfDataReader.h"

using namespace std;

/**
 * Empty constructor.
 */
MfDataReader::MfDataReader()
{
    wholeX = nullptr;
    wholeY = nullptr;
    trainingX = nullptr;
    trainingY = nullptr;
    testingX = nullptr;
    testingY = nullptr;
    randomArray = nullptr;
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
        printf("file not found\r\n");
        throw "file not found";
    }//Of if

    double tempDouble;
    int tempInt;
    char *tempValues;
    const char * tempSplit = ",";

    //The class value boundaries.
    int tempMinClass = 100;
    int tempMaxClass = -1;

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
                wholeX->setValue(tempInstanceIndex, i, tempDouble);
                //wholeX[0](tempInstanceIndex, i) = tempDouble;
                tempRemaining = strtok(NULL, tempSplit);
            }//Of for i

            sscanf(tempRemaining, "%d", &tempInt);
            //printf("instance %d, y = %d.\r\n", tempInstanceIndex, tempInt);
            wholeY->setValue(tempInstanceIndex, tempInt);
            //sscanf(tempRemaining, "%d", &[0](0, tempInstanceIndex));

            if (tempMinClass > tempInt)
            {
                tempMinClass = tempInt;
            }//Of if
            if (tempMaxClass < tempInt)
            {
                tempMaxClass = tempInt;
            }//Of if

            tempInstanceIndex ++;
        }//Of if
    }//Of while

    //printf("File read, wholeY is:\r\n");
    //cout << wholeY->toString() << endl;

    tempInputStream.close();
    numClasses = tempMaxClass - tempMinClass + 1;

    //Step 5. Maybe you do not want to randomize
    //printf("MfDataReader constructor test 5\r\n");
    randomArray = new MfIntArray(numInstances);
    for(int i = 0; i < numInstances; i ++)
    {
        randomArray->setValue(i, i);
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
    if (wholeX != nullptr)
    {
        printf("deleting wholeX\r\n");
        delete wholeX;
    }//Of if

    if (wholeY != nullptr)
    {
        printf("deleting wholeY\r\n");
        delete wholeY;
    }//Of if

    if (randomArray != nullptr)
    {
        printf("deleting randomArray\r\n");
        delete randomArray;
    }//Of if

    if (trainingX != nullptr)
    {
        printf("deleting trainingX\r\n");
        delete trainingX;
    }//Of if

    if (trainingY != nullptr)
    {
        printf("deleting trainingY\r\n");
        delete trainingY;
    }//Of if

    if (testingX != nullptr)
    {
        printf("deleting testingX\r\n");
        delete testingX;
    }//Of if

    if (testingY != nullptr)
    {
        printf("deleting testingX\r\n");
        delete testingY;
    }//Of if
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
        delete trainingX;
        trainingX = nullptr;
        delete trainingY;
        trainingY = nullptr;
        delete testingX;
        testingX = nullptr;
        delete testingY;
        testingY = nullptr;
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
            trainingX->setValue(i, j, wholeX->getValue(randomArray->getValue(i), j));
        }//Of for j
        //trainingY[0](0, i) = wholeY[0](0, randomArray[0](0, i));
        trainingY->setValue(i, wholeY->getValue(randomArray->getValue(i)));
    }//Of for i
    //printf("In MfDataReader, the training Y is:\r\n");
    //cout << trainingY->toString() << endl;

    //Testing set
    for(int i = tempTrainingSize; i < numInstances; i ++)
    {
        for(int j = 0; j < numConditions; j ++)
        {
            //testingX[0](i - tempTrainingSize, j) = wholeX[0](randomArray[0](0, i), j);
            testingX->setValue(i - tempTrainingSize, j, wholeX->getValue(randomArray->getValue(i), j));
        }//Of for j
        //testingY[0](0, i - tempTrainingSize) = wholeY[0](0, randomArray[0](0, i));
        testingY->setValue(i - tempTrainingSize, wholeY->getValue(randomArray->getValue(i)));
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
                trainingX->setValue(tempTrainingIndex, j, wholeX->getValue(randomArray->getValue(i), j));
            }//Of for j
            //trainingY[0](0, tempTrainingIndex) = wholeY[0](0, randomArray[0](0, i));
            trainingY->setValue(tempTrainingIndex, wholeY->getValue(randomArray->getValue(i)));
            tempTrainingIndex ++;
        }
        else
        {
            for(int j = 0; j < numConditions; j ++)
            {
                //testingX[0](tempTestingIndex, j) = wholeX[0](randomArray[0](0, i), j);
                testingX->setValue(tempTestingIndex, j, wholeX->getValue(randomArray->getValue(i), j));
            }//Of for j
            //testingY[0](0, tempTestingIndex) = wholeY[0](0, randomArray[0](0, i));
            testingY->setValue(tempTestingIndex, wholeY->getValue(randomArray->getValue(i)));
            tempTestingIndex ++;
        }//Of if
    }//Of for i
}//Of crossValidationSplit

/**
 * Randomize the data through generating a random int array.
 *   Data are accessed through indirect addressing.
 */
void MfDataReader::randomize()
{
    randomArray->randomizeOrder();
}//Of randomize

/**
 * Merge the data file of the same folder.
 * @param paraFoldername The folder name.
 * @param paraSuffix The suffix.
 */
void MfDataReader::fileMerge(char* paraFoldername, char* paraSuffix)
{
    string tempFolderString(paraFoldername);
    //string tempClass = to_string(paraClass);
    long tempFile;
    _finddata_t findFile;
    _chdir(paraFoldername);
    string tempFilename;

    if((tempFile=_findfirst(paraSuffix, &findFile))==-1L)

    {
        printf("No file exists.\n");
        exit(0);
    }//Of if

    FILE *tempOutFile;
    //string tempString(paraFileFolder);
    //tempString += "alldata.txt";
    if((tempOutFile = fopen((tempFolderString + "merged.data").data(), "w")) == NULL)
    {
        printf("Could not open file for writing.\r\n");
        exit(1);
    }//Of if

    string tempInputFileName = findFile.name;
    ifstream tempInputStream(tempInputFileName);
    string tempLine;
    printf("processing %s.\r\n", tempInputFileName.data());
    if (!tempInputStream)
    {
        printf("file not found\r\n");
        throw "file not found";
    }//Of if
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        fprintf(tempOutFile, "%s\r\n", tempLine.c_str());
    }//Of while
    tempInputStream.close();

    while(_findnext(tempFile, &findFile)==0)
    {
        tempInputFileName = findFile.name;
        printf("processing %s\r\n", tempInputFileName.c_str());
        tempInputStream.open(tempInputFileName);
        if (!tempInputStream)
        {
            printf("file not found\r\n");
            throw "file not found";
        }//Of if
        while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
        {
            fprintf(tempOutFile, "%s\r\n", tempLine.c_str());
        }//Of while
        tempInputStream.close();
    }//Of while _findnext

    fclose(tempOutFile);
}//Of fileMerge

/**
 * File merge test.
 */
void MfDataReader::fileMergeTest()
{
    string tempFoldername = "e:\\data\\petroleum\\pump\\train\\60times60\\";
    string tempSuffix = "*.txt";
    fileMerge((char *)tempFoldername.c_str(),
              (char *)tempSuffix.c_str());
}//Of fileMergeTest

/**
 * Code unit test.
 */
void MfDataReader::unitTest()
{
    string tempString = "d:\\c\\cann\\data\\iris.txt";
    char *tempFilename = (char *)tempString.c_str();
    //char *s_input = (char *)tempString.c_str();

    MfDataReader tempReader(tempFilename);

    printf("%d instances, %d conditions, %d classes\r\n", tempReader.numInstances,
           tempReader.numConditions, tempReader.numClasses);
    //free(tempFilename);

    //printf("Before randomize \r\n");
    tempReader.randomize();
    //printf("After ranomize \r\n");
    tempReader.splitInTwo(0.6);

    printf("The training X is: \r\n");
    cout << tempReader.trainingX->toString() << endl;
    //printf("The training X is: \r\n");
    printf("End of unit test. \r\n");

    //cout << tempArrayPtr[0] << endl;
}//Of unitTest
