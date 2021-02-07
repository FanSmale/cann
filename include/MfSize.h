/*
 * The C++ Artificial Neural network project.
 * Two dimensional size.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFSIZE_H
#define MFSIZE_H

#include <string>
using namespace std;

class MfSize
{
    public:
        //The default constructor.
        MfSize();

        //The constructor with enough parameters.
        MfSize(int, int);

        //The destructor.
        virtual ~MfSize();

        //Set width and height.
        void setValues(int, int);

        //Clone the size to me.
        MfSize* cloneToMe(MfSize* paraFirstSize);

        //Divide two sizes, the result is stored to me.
        MfSize* divideToMe(MfSize* paraFirstSize, MfSize* paraSecondSize);

        //Subtract two sizes, and append a value on both directions. The result is stored to me.
        MfSize* subtractToMe(MfSize* paraFirstSize, MfSize* paraSecondSize, int paraAppend);

        //For display
        string toString();

        //Unit test.
        void unitTest();

        //The width.
        int width;

        //The height.
        int height;

    protected:

    private:
};

#endif // MFSIZE_H
