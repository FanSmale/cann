/*
 * The C++ Artificial Neural network project.
 * Read a coordinate map from a file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "CoordinateMap.h"

/**
 * The empty constructor.
 */
CoordinateMap::CoordinateMap()
{
    data = nullptr;
    bitMap = nullptr;
}//Of the first constructor

/**
 * Read the data from the given file.
 * paraFilename: the data filename.
 */
CoordinateMap::CoordinateMap(char* paraFilename)
{
    //Step 1. Initialize
    length = 0;
    ifstream tempInputStream(paraFilename);
    string tempLine;

    if(!tempInputStream)
    {
        printf("file not found\r\n");
        throw "file not found";
    }//Of if

    double tempDouble;

    const char * tempSplit = ",";

    for(int i = 0; i < 2; i ++)
    {
        minValues[i] = 10000;
        maxValues[i] = -10000;
    }

    //Step 2. Prepare to read data. How many data points.
    //printf("MfDataReader constructor test 2\r\n");
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        if (tempLine.erase(0, tempLine.find_first_not_of(" ")) != "")
        {
            length ++;
        }//Of if
    }//Of while
    length --;

    //Step 3. Allocate space
    //printf("MfDataReader constructor test 3\r\n");
    printf("Data read, length = %d\r\n", length);

    data = (double**)malloc(length * sizeof(double*));
    for(int i = 0; i < length; i ++)
    {
        data[i] = (double*)malloc(2 * sizeof(double));
    }//Of for i

    //Step 4. Now read data
    tempInputStream.clear();
    tempInputStream.seekg(0, ios::beg);

    //Ignore the header
    getline(tempInputStream, tempLine);

    int tempRow = 0;
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        if (tempLine.erase(0, tempLine.find_first_not_of(" ")) != "")
        {
            stringstream tempStringStream(tempLine);
            string tempBuf;
            char *tempRemaining = (char *)tempLine.c_str();
            // Separated by comma
            tempRemaining = strtok(tempRemaining, tempSplit);
            for(int i = 0; i < 2; i ++)
            {
                sscanf(tempRemaining, "%lf", &tempDouble);
                //printf("MfDataReader constructor test 4.1.1, %lf\r\n", tempDouble);
                data[tempRow][i] = tempDouble;
                //wholeX[0](tempInstanceIndex, i) = tempDouble;
                tempRemaining = strtok(NULL, tempSplit);

                if (minValues[i] > tempDouble)
                {
                    minValues[i] = tempDouble;
                }//Of if
                if (maxValues[i] < tempDouble)
                {
                    maxValues[i] = tempDouble;
                }
            }//Of for i

            tempRow ++;
        }//Of if
    }//Of while
}//Of the second constructor

/**
 * The destructor.
 */
CoordinateMap::~CoordinateMap()
{
    if(data != nullptr)
    {
        for(int i = 0; i < length; i ++)
        {
            free(data[i]);
            data[i] = nullptr;
        }//Of for i
        free(data);
    }//Of if

    if(bitMap != nullptr)
    {
        for(int i = 0; i < height; i ++)
        {
            free(bitMap[i]);
            bitMap[i] = nullptr;
        }//Of for i
        free(bitMap);
    }//Of if
}//Of the destructor

/**
 * Convert to string for display.
 */
string CoordinateMap::toString()
{
    string resultString = "Original data\r\n";
    for(int i = 0; i < length; i ++)
    {
        for(int j = 0; j < 2; j ++)
        {
            resultString += to_string(data[i][j]) +  ", ";
        }//Of for j
        resultString += "\r\n";
    }//Of for i

    resultString += "Bit map\r\n";
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j ++)
        {
            resultString += to_string(bitMap[i][j]) +  " ";
        }//Of for j
        resultString += "\r\n";
    }//Of for i
    resultString += "Map ends. \r\n";

    return resultString;
}//Of toString

/**
 * Construct the bit map.
 * paraHeight The height of the map.
 * paraWidth The width of the map.
 */
void CoordinateMap::constructBitMap(int paraHeight, int paraWidth)
{
    //printf("constructBitMap test 1\r\n");
    height = paraHeight;
    width = paraWidth;

    bitMap = (int**)malloc(height * sizeof(int*));
    for(int i = 0; i < height; i ++)
    {
        bitMap[i] = (int*)malloc(width * sizeof(int));
        for(int j = 0; j < width; j ++)
        {
            bitMap[i][j] = 0;
        }//Of for j
    }//Of for i

    //printf("constructBitMap test 2\r\n");
    //Assign for each data point.
    int tempX;
    int tempY;
    double tempXDifference = maxValues[0] - minValues[0] + 0.01;
    double tempYDifference = maxValues[1] - minValues[1] + 0.01;

    for(int i = 0; i < length; i ++)
    {
        tempX = (int)((data[i][0] - minValues[0]) / tempXDifference * width);
        tempY = (int)((data[i][1] - minValues[1]) / tempYDifference * height);
        //printf("tempX = %d, tempY = %d\r\n", tempX, tempY);
        bitMap[tempY][tempX] = 1;
    }//Of for i

    //printf("constructBitMap test end\r\n");
}//Of constructBitMap

/**
 * Convert the bit map to string.
 */
string CoordinateMap::bitMapToString()
{
    string resultString = "";
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j ++)
        {
            resultString += to_string(bitMap[i][j]) +  ",";
        }//Of for j
    }//Of for i

    return resultString;
}//Of bitMapToString

/**
 * Code unit test.
 */
void CoordinateMap::unitTest()
{
    string tempString = "e:\\data\\petroleum\\pump\\train\\A01\\A01_136214_0.csv";
    char *tempFilename = (char *)tempString.c_str();
    //char *s_input = (char *)tempString.c_str();

    CoordinateMap* tempMap = new CoordinateMap(tempFilename);
    tempMap -> constructBitMap(18, 36);

    printf(tempMap->toString().data());

    printf(tempMap->bitMapToString().data());
}//Of unitTest
