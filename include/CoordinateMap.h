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


class CoordinateMap
{
    public:
        CoordinateMap();

        //Read a map from a file.
        CoordinateMap(char* paraFilename);

        virtual ~CoordinateMap();

        //Convert to string for display.
        string toString();

    protected:

    private:
        //The number of data points.
        double length;

        //The data.
        double*** data;

        //The bit map.
        bool*** bitMap;

        //The bit map in string.
        bitMapInString;

        //The minimal values of each coordinate
        double minValues[2];

        //The maximal values of each coordinate
        double maxValues[2];
};

#endif // COORDINATEMAP_H
