

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

string ExtractDirectory( const std::string& path );

string ExtractFilename( const std::string& path );

string ChangeExtension( const std::string& path, const std::string& ext );

string makeVocabularyFileName(const string& impath, int vocSize);

bool readVocabulary( const string& filename, Mat& vocabulary );

bool writeVocabulary( const string& filename, Mat& vocabulary);
