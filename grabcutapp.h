
#ifndef __GRABCUTAPP_H__
#define __GRABCUTAPP_H__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "grabcutapp.h"
#include <iostream>

using namespace std;
using namespace cv;

int grabcutapp(string filename, Mat& finalmask);

#endif