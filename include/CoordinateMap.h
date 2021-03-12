/*
 * The C++ Artificial Neural network project.
 * Read a coordinate map from a file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef COORDINATEMAP_H
#define COORDINATEMAP_H

//#include <string>
#include <string.h>
#include <Malloc.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

class CoordinateMap
{
    public:
        //The empty constructor.
        CoordinateMap();

        //Read a map from a file.
        CoordinateMap(char* paraFilename);

        //The destructor.
        virtual ~CoordinateMap();

        //Convert to string for display.
        string toString();

        //Convert to a bit map.
        void constructBitMap(int paraHeight, int paraWidth);

        //Convert the bitmap to string.
        string bitMapToString();

        //Code unit test
        void unitTest();

    protected:

    private:
        //The number of data points.
        int length;

        //The data.
        double** data;

        //The height of the bitmap.
        int height;

        //The width of the bitmap.
        int width;

        //The bit map.
        int** bitMap;

        //The bit map in string.
        string bitMapInString;

        //The minimal values of each coordinate
        double minValues[2];

        //The maximal values of each coordinate
        double maxValues[2];
};

#endif // COORDINATEMAP_H
