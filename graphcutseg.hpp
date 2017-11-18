
#ifndef __GRAPHCUTSEG_HPP__
#define __GRAPHCUTSEG_HPP__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "FastGauss.h"

#include "gcgraph.hpp"
#include <limits>
#include <iostream>

using namespace std;


/*! \namespace cv
 Namespace where all the C++ OpenCV functionality resides
 */
using namespace cv;



void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma );


/*
  Estimate segmentation using MaxFlow algorithm
*/
void estimateSegmentation( GCGraph<double>& graph, Mat& mask );

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
double calcBeta( const Mat& img );



#endif