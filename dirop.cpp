


#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

string ExtractDirectory( const std::string& path )
{
	return path.substr( 0, path.find_last_of( '/' ) +1 );
}

string ExtractFilename( const std::string& path )
{
	return path.substr( path.find_last_of( '/' ) +1 );
}

string ChangeExtension( const std::string& path, const std::string& ext )
{
	string filename = ExtractFilename( path );
	return ExtractDirectory( path ) +filename.substr( 0, filename.find_last_of( '.' ) ) +ext;
}

string makeVocabularyFileName(const string& impath, int vocSize)
{
	char ext[20];
	string filename = ExtractFilename(impath);
	sprintf(ext,"_voc_%i.db",vocSize);
	return ExtractDirectory(impath)+filename.substr( 0, filename.find_last_of( '.' ) )+	ext;
}


bool readVocabulary( const string& filename, Mat& vocabulary )
{
    FileStorage fs( filename, FileStorage::READ );
    //cout << "Reading vocabulary...";
	
    if( fs.isOpened() )
    {
        fs["vocabulary"] >> vocabulary;
        //cout << "done" << endl;
        return true;
    }
    return false;
}

bool writeVocabulary( const string& filename, Mat& vocabulary)
{
    //cout << "Saving vocabulary..." << endl;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}
