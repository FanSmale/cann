/*
 * The C++ Artificial Neural network project.
 * Convert a number of bit map files into one file.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFBITMAPSCONVERTER_H
#define MFBITMAPSCONVERTER_H

#include <string.h>
#include <Malloc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <io.h>
#include <dir.h>
#include <dirent.h>
#include <direct.h>

#include "CoordinateMap.h"

using namespace std;

class MfBitMapsConverter
{
    public:
        //The empty constructor.
        MfBitMapsConverter();

        //Read maps from a file folder.
        MfBitMapsConverter(char* paraFileFolder, int paraClass, int paraHeight, int paraWidth);

        //The destructor.
        virtual ~MfBitMapsConverter();

        //Convert to string for display.
        string toString();

        //Code unit test
        void unitTest();

    protected:

    private:
};

#endif // MFBITMAPSCONVERTER_H
