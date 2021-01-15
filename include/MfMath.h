#ifndef MFMATH_H_INCLUDED
#define MFMATH_H_INCLUDED

#include <stdlib.h>
#include <D:\c\eigen\Eigen\Dense>
#include <D:\c\eigen\Eigen\src\core\Array.h>

//Constants
#define OUT_OF_RANGE_EXCEPTION -1
#define LENGTH_NOT_MATCH_EXCEPTION -3

//Classes
//#define DoubleMatrix Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
//#define DoubleArray Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>
#define DoubleMatrix Matrix<double, Dynamic, Dynamic>
#define DoubleArray Array<double, Dynamic, Dynamic>
//Row array
#define IntArray Matrix<int, 1, Dynamic>

#define random() (rand()+0.0)/RAND_MAX

using namespace Eigen;

#endif // MFMATH_H_INCLUDED
