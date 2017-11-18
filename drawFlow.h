

#ifndef __DRAWFLOW__
#define __DRAWFLOW__

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double scale, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+scale*fxy.x), cvRound(y+scale*fxy.y)),
                 color);
            circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
}
#endif