/*
 * The C++ Artificial Neural network project.
 * This class manages matrices.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MF_MATRIX_H
#define MF_MATRIX_H

#include <string>

using namespace std;

class MfMatrix
{
    public:

        //The default constructor
        MfMatrix();

        //Initialize a matrix with given sizes
        MfMatrix(int paraRows, int paraColumns);

        //Destructor
        virtual ~MfMatrix();

        //Convert to string for display
        string toString();

        //Get a value at the given position
        double getValue(int paraRow, int paraColumn);

        //Set a value at the given position
        double setValue(int paraRow, int paraColumn, double paraValue);

        //Copy a matrix
        MfMatrix* copy();

        //Add another one with the same size
        MfMatrix* add(MfMatrix* paraMfMatrix);

        //Add another one with the same size to me, no space allocation
        void addToMe(MfMatrix* paraMfMatrix);

        //Minus another one with the same size
        MfMatrix* minus(MfMatrix* paraMfMatrix);

        //Minus another one with the same size to me, no space allocation
        void minusToMe(MfMatrix* paraMfMatrix);

        //Multiply another one with the same size
        MfMatrix* multiply(MfMatrix* paraMfMatrix);

        //Multiply another one with the same size to me, no space allocation
        void multiplyToMe(MfMatrix* paraMfMatrix);

        //Dot multiply, return a new matrix
        MfMatrix* dot(MfMatrix* paraMfMatrix);

        //Transpose, return a new matrix
        MfMatrix* transpose();

        //Activate all values of the matrix
        void activate(char paraFunction);

        //Activate
        double activate(double paraValue, char paraFunction);

        //Code unit test
        void selfTest();

    protected:

        //Number of rows
        int rows;

        //Number of columns
        int columns;

        //The data
        double** data;

        //string tempString;

    private:
};

#endif // MATRIX_H
