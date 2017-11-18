
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "graphcutseg.hpp"
#include "gcgraph.hpp"
#include <limits>
#include <iostream>

using namespace std;


/*! \namespace cv
 Namespace where all the C++ OpenCV functionality resides
 */
using namespace cv;



//! Class of the pixel 
enum { GC_BGD_MASK    = 2,  //!< background
		GC_FGD_MASK    = 1,  //!< foreground
		};

void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3d>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols-1 && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}


/*
  Estimate segmentation using MaxFlow algorithm
*/
void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    cv::Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {

            if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                mask.at<uchar>(p) = 1;
            else
                mask.at<uchar>(p) = 0;
            
        }
    }
}

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3d>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3d>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    return beta;
}

