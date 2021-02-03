/*
 * The C++ Artificial Neural network project.
 * This class manages double matrices.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <MfDoubleMatrix.h>

using namespace std;

/**
 * The default constructor.
 */
MfDoubleMatrix::MfDoubleMatrix()
{
    rows = 0;
    columns = 0;
    data = nullptr;
}//Of the default constructor

//
/**
 * The constructor. Initialize a matrix with given sizes.
 * paraRows: the number of rows.
 * paraColumns: the number of columns.
 */
MfDoubleMatrix::MfDoubleMatrix(int paraRows, int paraColumns)
{
    rows = paraRows;
    columns = paraColumns;

    //Allocate space
    data = new double *[rows];
    for (int i = 0; i < rows; i ++)
    {
        data[i] = new double[columns];
    }//Of for i

    //Some initial values
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = rand() / (double)(RAND_MAX);
        }//Of for j
    }//Of for i
}//Of the second constructor

//Destructor
MfDoubleMatrix::~MfDoubleMatrix()
{
    for (int i = 0; i < rows; i ++)
    {
        free(data[i]);
    }//Of for i
    free(data);
}//Of the destructor

//Show me with a string for display
string MfDoubleMatrix::toString()
{
    //string resultString = "I am a matrix with size " + to_string(rows)
    //    + "*" + to_string(columns) + "\r\n";
    string resultString = "";
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultString += to_string(data[i][j]) +  ", ";
        }//Of for j
        resultString += "\r\n";
    }//Of for i

    //resultString += "matrix ends. \r\n";

    return resultString;
}//Of toString

/**
 * Getter.
 */
int MfDoubleMatrix::getRows()
{
    return rows;
}//Of getRows

/**
 * Getter.
 */
int MfDoubleMatrix::getColumns()
{
    return columns;
}//Of getColumns

/**
 * Getter. Please use with caution because it is a pointer.
 */
double** MfDoubleMatrix::getData()
{
    return data;
}//Of getData

/**
 * Getter. Get a value at the given position.
 * paraRow: the row.
 * paraColumn: the column.
 */
double MfDoubleMatrix::getValue(int paraRow, int paraColumn)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("MfDoubleMatrix.getValue() out of range.");
        throw "MfDoubleMatrix.getValue() out of range.";
    }//Of if
    return data[paraRow][paraColumn];
}//Of getValue

/**
 * Setter. Set a value at the given position.
 * paraRow: the row.
 * paraColumn: the column.
 * paraValue: the new value.
 */
double MfDoubleMatrix::setValue(int paraRow, int paraColumn, double paraValue)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("MfDoubleMatrix.setValue() out of range.");
        throw "MfDoubleMatrix.setValue() out of range.";
    }//Of if
    data[paraRow][paraColumn] = paraValue;

    return paraValue;
}//Of setValue

/**
 * Copy a matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::copy()
{
    MfDoubleMatrix* resultMfDoubleMatrix = new MfDoubleMatrix(rows, columns);
    double** tempData = resultMfDoubleMatrix -> data;

    //Copy
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] = data[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of copy

/**
 * Copy from a matrix. Change my values.
 * paraMatrix: the given matrix
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::copyFrom(MfDoubleMatrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    //Copy
    double** tempData = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = tempData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of copyFrom

/**
 * Add another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::add(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    //Maybe the runtime can be saved in this way.
    double** tempData = resultMfDoubleMatrix -> data;
    double** tempData2 = paraMatrix -> data;

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] += tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of add

/**
 * Add another one with the same size to me.
 * paraMatrix: the other matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::addToMe(MfDoubleMatrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("MfDoubleMatrix.addToMe(), rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("MfDoubleMatrix.addToMe(), columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempData = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] += tempData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of addToMe

/**
 * Minus another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::minus(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("MfDoubleMatrix.minus(), rows do not match.");
        throw "Rows do not match";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("MfDoubleMatrix.minus(), columns do not match.");
        throw "Columns do not match";
    }//Of if

    double** tempData = resultMfDoubleMatrix -> data;
    double** tempData2 = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] -= tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of minus

/**
 * Minus another one with the same size to me.
 * paraMatrix: the other matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::minusToMe(MfDoubleMatrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempData = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] -= tempData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of minusToMe

/**
 * Multiply another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::cwiseProduct(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = copy();

    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempData = resultMfDoubleMatrix -> data;
    double** tempData2 = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] *= tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of cwiseProduct

/**
 * Multiply another one with the same size to me.
 * paraMatrix: the other matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::cwiseProductToMe(MfDoubleMatrix* paraMatrix)
{
    if (rows != paraMatrix -> rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix -> columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempData = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] *= tempData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of cwiseProductToMe

/**
 * Times another one.
 * paraMatrix: the other matrix with size n*k.
 * Return: a new matrix with size m*k.
 */
MfDoubleMatrix* MfDoubleMatrix::times(MfDoubleMatrix *paraMatrix)
{
    int tempColumns = paraMatrix -> getColumns();

    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns ,tempColumns);
    if (columns != paraMatrix -> rows)
    {
        printf("Matrices do not match.");
        throw "Matrices do not match.";
    }//Of if

    //Attention: one exception happened here consistently, but now disappeared.
    //That is, columns changed by the following statement.
    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns,tempColumns);
    MfDoubleMatrix* newMfDoubleMatrixPtr = new MfDoubleMatrix(rows, tempColumns);
    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns, tempColumns);
    double tempValue = 0;

    double** tempData1 = newMfDoubleMatrixPtr -> data;
    double** tempData2 = paraMatrix -> data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < tempColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < columns; k ++)
            {
                tempValue += data[i][k] * tempData2[k][j];
            }//Of for k
            tempData1[i][j] = tempValue;
        }//Of for j
    }//Of for i

    return newMfDoubleMatrixPtr;
}//Of times

/**
 * Two matrices times to fill me.
 * paraFirstMatrix: the first matrix, size m*n.
 * paraSecondMatrix: the second matrix, size n*k.
 * Return: myself, size m*k.
 */
MfDoubleMatrix* MfDoubleMatrix::timesToMe(MfDoubleMatrix *paraFirstMatrix, MfDoubleMatrix *paraSecondMatrix)
{
    int tempFirstRows = paraFirstMatrix -> getRows();
    int tempFirstColumns = paraFirstMatrix -> getColumns();
    int tempSecondRows = paraSecondMatrix -> getRows();
    int tempSecondColumns = paraSecondMatrix -> getColumns();

    double** tempFirstData = paraFirstMatrix -> data;
    double** tempSecondData = paraSecondMatrix -> data;

    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns ,tempColumns);
    if (tempFirstColumns != tempSecondRows)
    {
        printf("Matrices do not match.");
        throw "Matrices do not match.";
    }//Of if

    double tempValue = 0;

    for (int i = 0; i < tempFirstRows; i ++)
    {
        for (int j = 0; j < tempSecondColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < tempFirstColumns; k ++)
            {
                tempValue += tempFirstData[i][k] * tempSecondData[k][j];
            }//Of for k
            data[i][j] = tempValue;
        }//Of for j
    }//Of for i

    return this;
}//Of timesToMe

/**
 * Transpose the matrix to another.
 * Return: the transposed matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::transpose()
{
    MfDoubleMatrix* newMfDoubleMatrixPtr = new MfDoubleMatrix(columns, rows);

    double** tempData = newMfDoubleMatrixPtr -> data;
    for (int i = 0; i < columns; i ++)
    {
        for (int j = 0; j < rows; j ++)
        {
            tempData[i][j] = data[j][i];
        }//Of for j
    }//Of for i

    return newMfDoubleMatrixPtr;
}//Of transpose

/**
 * Fill the matrix with the same value.
 * paraValue: the given value.
 */
void MfDoubleMatrix::fill(double paraValue)
{
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = paraValue;
        }//Of for j
    }//Of for i
}//Of fill

/**
 * Code unit test.
 */
void MfDoubleMatrix::unitTest()
{
    MfDoubleMatrix* tempMfDoubleMatrix = new MfDoubleMatrix(2, 3);
    printf("Original\r\n");

    printf(tempMfDoubleMatrix -> toString().data());

    MfDoubleMatrix* tempMfDoubleMatrix2 = tempMfDoubleMatrix -> copy();
    printf("Copy\r\n");
    printf(tempMfDoubleMatrix2 -> toString().data());

    MfDoubleMatrix* tempTransposed = tempMfDoubleMatrix -> transpose();
    printf("Transpose\r\n");
    printf(tempTransposed -> toString().data());

    MfDoubleMatrix* tempDot = tempMfDoubleMatrix -> times(tempTransposed);
    printf("Dot\r\n");
    printf(tempDot -> toString().data());

    MfDoubleMatrix* tempAdded = tempMfDoubleMatrix -> add(tempMfDoubleMatrix2);
    printf("Add\r\n");
    printf(tempAdded -> toString().data());

    MfDoubleMatrix* tempMultiply = tempMfDoubleMatrix -> cwiseProduct(tempMfDoubleMatrix2);
    printf("Multiply\r\n");
    printf(tempMultiply -> toString().data());

    MfDoubleMatrix* tempMinus = tempMultiply  -> minus(tempMfDoubleMatrix);
    printf("Minus\r\n");
    printf(tempMinus -> toString().data());
}//Of unitTest
