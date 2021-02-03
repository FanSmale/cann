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

#ifndef MFOUBLEMATRIX_H
#define MFOUBLEMATRIX_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

#include "Malloc.h"

using namespace std;

class MfDoubleMatrix
{
    public:

        //The default constructor
        MfDoubleMatrix();

        //Initialize a matrix with given sizes
        MfDoubleMatrix(int paraRows, int paraColumns);

        //Destructor
        virtual ~MfDoubleMatrix();

        //Convert to string for display
        string toString();

        //Get a value at the given position
        double getValue(int paraRow, int paraColumn);

        //Getter
        int getRows();

        //Getter
        int getColumns();

        //Getter
        double** getData();

        //Set a value at the given position
        double setValue(int paraRow, int paraColumn, double paraValue);

        //Copy a matrix
        MfDoubleMatrix* copy();

        //Copy a matrix
        MfDoubleMatrix* copyFrom(MfDoubleMatrix*);

        //Add another one with the same size
        MfDoubleMatrix* add(MfDoubleMatrix* paraMfDoubleMatrix);

        //Add another one with the same size to me, no space allocation
        MfDoubleMatrix* addToMe(MfDoubleMatrix* paraMfDoubleMatrix);

        //Minus another one with the same size
        MfDoubleMatrix* minus(MfDoubleMatrix* paraMfDoubleMatrix);

        //Minus another one with the same size to me, no space allocation
        MfDoubleMatrix* minusToMe(MfDoubleMatrix* paraMfDoubleMatrix);

        //Point-to-point multiply another one with the same size
        MfDoubleMatrix* cwiseProduct(MfDoubleMatrix* paraMfDoubleMatrix);

        //Point-to-point multiply another one with the same size to me, no space allocation
        MfDoubleMatrix* cwiseProductToMe(MfDoubleMatrix* paraMfDoubleMatrix);

        //Times another one, return a new matrix, m*n times n*k gets m*k
        MfDoubleMatrix* times(MfDoubleMatrix* paraMfDoubleMatrix);

        //Times to me, return myself, m*n times n*k gets m*k
        MfDoubleMatrix* timesToMe(MfDoubleMatrix *paraFirstMatrix, MfDoubleMatrix *paraSecondMatrix);

        //Transpose, return a new matrix
        MfDoubleMatrix* transpose();

        //Fill the matrix with the same value
        void fill(double paraValue);

        //Code unit test
        void unitTest();

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

#endif // MFOUBLEMATRIX_H
