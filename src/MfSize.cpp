/*
 * The C++ Artificial Neural network project.
 * Two dimensional size.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include "MfSize.h"

/**
 * The default constructor.
 */
MfSize::MfSize()
{
    width = 0;
    height = 0;
}//Of the default constructor

/**
 * The constructor with enough parameters.
 */
MfSize::MfSize(int paraWidth, int paraHeight)
{
    width = paraWidth;
    height = paraHeight;
}//Of the second constructor

/**
 * The destructor.
 */
MfSize::~MfSize()
{
    //dtor
}//Of the destructor

/**
 * Clone the size to me.
 * paraSize: the given size.
 * Return: myself.
 */
MfSize* MfSize::cloneToMe(MfSize* paraSize)
{
    width = paraSize->width;
    height = paraSize->height;

    return this;
}//Of cloneToMe

/**
 * Divide two sizes, the result store to me.
 * paraFirstSize: the first size.
 * paraSecondSize: the second size.
 * Return: myself.
 */
MfSize* MfSize::divideToMe(MfSize* paraFirstSize, MfSize* paraSecondSize)
{
    width = paraFirstSize->width / paraSecondSize->width;
    height = paraFirstSize->height / paraSecondSize->height;
    if (width * paraSecondSize->width != paraFirstSize->width)
    {
        printf("MfSize::divideToMe(), the width cannot divide.");
        throw "MfSize::divideToMe(), the width cannot divide.";
    }//Of if

    if (height * paraSecondSize->height != paraFirstSize->height )
    {
        printf("MfSize::divideToMe(), the width cannot divide.");
        throw "MfSize::divideToMe(), the width cannot divide.";
    }//Of if

    return this;
}//Of divideToMe

/**
 * Subtract a scale with another one, and add a value. For example (4, 12) - (2, 3) + 1 = (3, 10).
 * paraFirstSize: the first size.
 * paraSecondSize: the second size.
 * paraAppend: the appended value to both width and height.
 * Return: myself.
 */
MfSize* MfSize::subtractToMe(MfSize* paraFirstSize, MfSize* paraSecondSize, int paraAppend)
{
    width  = paraFirstSize->width - paraSecondSize->width + paraAppend;
    height = paraFirstSize->height -  paraSecondSize->height + paraAppend;
    if ((width < 1) || (height < 1))
    {
        printf("MfSize::subtractToMe, the new size is less than 1.");
        throw "MfSize::subtractToMe, the new size is less than 1.";
    }//Of if

    return this;
}//Of subtractToMe

/**
 * Convert to string for display.
 * Returns: The string showing myself.
 */
string MfSize::toString()
{
    string resultString = "(" + to_string(width)
                          + ", " + to_string(height) + ")";

    return resultString;
}//Of toString

/**
 * Unit test.
 */
void MfSize::unitTest()
{
    MfSize* tempFirstSize = new MfSize(10, 6);
    MfSize* tempSecondSize = new MfSize(2, 3);
    MfSize* tempThirdSize = new MfSize();
    tempThirdSize->divideToMe(tempFirstSize, tempSecondSize);
    printf("Divide result: %s\r\n", tempThirdSize->toString().data());

    tempThirdSize->subtractToMe(tempFirstSize, tempSecondSize, 3);
    printf("Subtract result: %s\r\n", tempThirdSize->toString().data());

    //tempThirdSize->subtractToMe(tempFirstSize, tempSecondSize, -10);
    //printf("Subtract result: %s\r\n", tempThirdSize->toString().data());

}//Of unit test
