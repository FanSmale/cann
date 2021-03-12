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

/*
 * The empty constructor.
 */
CoordinateMap::CoordinateMap()
{
    data = nullptr;
    //ctor
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

    if (!tempInputStream)
    {
        printf("file not found\r\n");
        throw "file not found";
    }//Of if

    double tempDouble;
    int tempInt;
    char *tempValues;
    const char * tempSplit = ",";

    for (int i = 0; i < 2; i ++)
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
    for(int i = 0; i < rows; i ++)
    {
        data[i] = (double*)malloc(2 * sizeof(double));
    }//Of for i

    //Step 4. Now read data
    tempInputStream.clear();
    tempInputStream.seekg(0, ios::beg);

    int tempInstanceIndex = 0;
    //Ignore the header
    getline(tempInputStream, tempLine);

    int tempRow = 0;
    while (getline(tempInputStream, tempLine)) // line中不包括每行的换行符
    {
        tempRow = 0;
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

CoordinateMap::~CoordinateMap()
{
    //dtor

}//Of the destructor

        //Convert to string for display.
        string toString();

