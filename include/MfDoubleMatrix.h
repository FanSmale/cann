/*
 * The C++ Artificial Neural network project.
 * This class manages matrices. To facilitate space management, I do not want to generate new matrices.
 *   So many methods are named as xxToMe(). In this way, the space should be allocated outside the functions.
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
#include <Malloc.h>

#include "Activator.h"
#include "MfSize.h"

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

        //Set the activator
        void setActivator(Activator* paraActivator);

        //Activate, return myself
        MfDoubleMatrix* activate();

        //Copy a matrix
        MfDoubleMatrix* clone();

        //Copy a matrix
        MfDoubleMatrix* cloneFrom(MfDoubleMatrix* paraMfDoubleMatrix);

        //Add another one with the same size
        MfDoubleMatrix* add(MfDoubleMatrix* paraMfDoubleMatrix);

        //Add another one with the same size to me, no space allocation
        MfDoubleMatrix* addToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix);

        //Minus another one with the same size
        MfDoubleMatrix* subtract(MfDoubleMatrix* paraMfDoubleMatrix);

        //Minus another one with the same size to me, no space allocation
        MfDoubleMatrix* subtractToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix);

        //Point-to-point multiply another one with the same size
        MfDoubleMatrix* cwiseProduct(MfDoubleMatrix* paraMfDoubleMatrix);

        //Point-to-point multiply another one with the same size to me, no space allocation
        MfDoubleMatrix* cwiseProductToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix);

        //Times another one, return a new matrix, m*n times n*k gets m*k
        MfDoubleMatrix* times(MfDoubleMatrix* paraMfDoubleMatrix);

        //Times to me, return myself, m*n times n*k gets m*k
        MfDoubleMatrix* timesToMe(MfDoubleMatrix *paraFirstMatrix, MfDoubleMatrix *paraSecondMatrix);

        //Transpose, return a new matrix
        MfDoubleMatrix* transpose();

        //Fill the matrix with the same value
        void fill(double paraValue);

        //Convolution valid, the size is smaller, return myself
        MfDoubleMatrix* convolutionValidToMe(MfDoubleMatrix *paraData, MfDoubleMatrix *paraKernel);

        //Convolution full, the size is unchanged, return myself
        MfDoubleMatrix* convolutionFullToMe(MfDoubleMatrix *paraData, MfDoubleMatrix *paraKernel);

        //Rotate 180 degrees.
        MfDoubleMatrix* rotate180();

        //Rotate 180 degrees.
        MfDoubleMatrix* rotate180ToMe(MfDoubleMatrix* paraMatrix);

        //Scale the matrix with the given size.
        MfDoubleMatrix* scale(MfSize* paraSize);

        //Scale the matrix with the given size to me.
        MfDoubleMatrix* scaleToMe(MfDoubleMatrix* paraMatrix, MfSize* paraSize);

        //Kronecker: copy many times.
        MfDoubleMatrix* kronecker(MfSize* paraSize);

        //Kronecker: copy many times.
        MfDoubleMatrix* kroneckerToMe(MfDoubleMatrix* paraMatrix, MfSize* paraSize);

        //Code unit test
        void unitTest();

    protected:

        //Number of rows
        int rows;

        //Number of columns
        int columns;

        //The data
        double** data;

        //The activator
        Activator* activator;

    private:

};

#endif // MFOUBLEMATRIX_H
