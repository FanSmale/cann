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

#ifndef MATRIX_H
#define MATRIX_H

#include <string>

using namespace std;

class Matrix
{
    public:

        //The default constructor
        Matrix();

        //Initialize a matrix with given sizes
        Matrix(int paraRows, int paraColumns);

        //Destructor
        virtual ~Matrix();

        //Convert to string for display
        string toString();

        //Get a value at the given position
        double getValue(int paraRow, int paraColumn);

        //Set a value at the given position
        double setValue(int paraRow, int paraColumn, double paraValue);

        //Copy a matrix
        Matrix* copy();

        //Add another one with the same size
        Matrix* add(Matrix* paraMatrix);

        //Add another one with the same size to me, no space allocation
        void addToMe(Matrix* paraMatrix);

        //Minus another one with the same size
        Matrix* minus(Matrix* paraMatrix);

        //Minus another one with the same size to me, no space allocation
        void minusToMe(Matrix* paraMatrix);

        //Multiply another one with the same size
        Matrix* multiply(Matrix* paraMatrix);

        //Multiply another one with the same size to me, no space allocation
        void multiplyToMe(Matrix* paraMatrix);

        //Dot multiply, return a new matrix
        Matrix* dot(Matrix* paraMatrix);

        //Transpose, return a new matrix
        Matrix* transpose();

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
