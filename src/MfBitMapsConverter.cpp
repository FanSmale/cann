#include "MfBitMapsConverter.h"

/**
 * The empty constructor.
 */
MfBitMapsConverter::MfBitMapsConverter()
{
    //ctor
}//Of the first constructor

/**
 * Read the data from the given folder.
 * paraFilename: the data folder name.
 */
MfBitMapsConverter::MfBitMapsConverter(char* paraFileFolder, int paraClass,
                                       int paraHeight, int paraWidth)
{
    string tempFolderString(paraFileFolder);
    string tempClass = to_string(paraClass);
    long tempFile;
    _finddata_t findFile;
    _chdir(paraFileFolder);
    string tempCsvFilename;

    if((tempFile=_findfirst("*.csv", &findFile))==-1L)

    {
        printf("No file exists.\n");
        exit(0);
    }//Of if

    FILE *tempOutFile;
    //string tempString(paraFileFolder);
    //tempString += "alldata.txt";
    if((tempOutFile = fopen((tempFolderString + tempClass + ".txt").data(), "w")) == NULL)
    {
        printf("Could not open file for writing.\r\n");
        exit(1);
    }//Of if

    tempCsvFilename = findFile.name;
    tempCsvFilename = tempFolderString + tempCsvFilename;
    //printf(tempCsvFilename.c_str());
    CoordinateMap* tempMap = new CoordinateMap((char *)tempCsvFilename.c_str());
    tempMap -> constructBitMap(paraHeight, paraWidth);
    string tempData = tempMap -> bitMapToString() + tempClass;
    fprintf(tempOutFile, "%s\r\n", tempData.c_str());

    while(_findnext(tempFile, &findFile)==0)
    {
        tempCsvFilename = findFile.name;
        tempCsvFilename = tempFolderString + tempCsvFilename;
        //printf("%s\n", tempCsvFilename.c_str());
        delete tempMap;
        tempMap = new CoordinateMap((char *)tempCsvFilename.c_str());
        tempMap -> constructBitMap(paraHeight, paraWidth);
        tempData = tempMap -> bitMapToString() + tempClass;
        fprintf(tempOutFile, "%s\r\n", tempData.c_str());
    }//Of while

    fclose(tempOutFile);
}//Of the second constructor

/**
 * The destructor.
 */
MfBitMapsConverter::~MfBitMapsConverter()
{
}//Of the destructor

/**
 * Convert to string for display.
 */
string MfBitMapsConverter::toString()
{
    string resultString = "MfBitMapsConverter\r\n";
    return resultString;
}//Of toString

/**
 * Code unit test.
 */
void MfBitMapsConverter::unitTest()
{
    int tempMaxIndex = 12;
    string tempString = "e:\\data\\petroleum\\pump\\train\\A";
    string tempNewString;
    char *tempFoldername;
    MfBitMapsConverter* tempConverter;
    for(int i = 1; i <= tempMaxIndex; i ++)
    {
        if (i < 10)
        {
            tempNewString = tempString + "0" + to_string(i) + "\\";
        } else
        {
            tempNewString = tempString + to_string(i) + "\\";
        }

        tempFoldername = (char *)tempNewString.c_str();
        tempConverter = new MfBitMapsConverter(tempFoldername, i, 60, 60);
        delete(tempConverter);
    }//Of for i

    //char *tempFoldername = (char *)tempNewString.c_str();
    //char *s_input = (char *)tempString.c_str();
}//Of unitTest
