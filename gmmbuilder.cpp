

#include "gmmbuilder.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

//#include "precomp.hpp"
#include <limits>

using namespace cv;

GMMBuilder::GMMBuilder(const Mat& _img, const Mat& _mask)
{

	if( _img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( _img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	img = _img.clone();
	mask = _mask.clone();

	Mat bgdModel, fgdModel;

	bgdGMM = GMM( bgdModel );
	fgdGMM = GMM( fgdModel );
    compIdxs = Mat( img.size(), CV_32SC1 );

	initGMMs();

	assignGMMsComponents();

	learnGMMs();


}


/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
void GMMBuilder::initGMMs()
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    cv::Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD_MASK)
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else if ( mask.at<uchar>(p) == GC_FGD_MASK)
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() || !fgdSamples.empty() );

	if( !bgdSamples.empty() )
	{
		Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
		kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
				TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, noArray() );

		bgdGMM.initLearning();
		for( int i = 0; i < (int)bgdSamples.size(); i++ )
			bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
		bgdGMM.endLearning();

	}
	if( !fgdSamples.empty() )
	{
		Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
		kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
				TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType,noArray() );

		fgdGMM.initLearning();
		for( int i = 0; i < (int)fgdSamples.size(); i++ )
			fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
		fgdGMM.endLearning();
	}
}

/*
  Assign GMMs components for each pixel.
*/
void GMMBuilder::assignGMMsComponents()
{
    cv::Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
			if(mask.at<uchar>(p) == GC_BGD_MASK)
				compIdxs.at<int>(p) = bgdGMM.whichComponent(color);
			else if(mask.at<uchar>(p) == GC_FGD_MASK)
				compIdxs.at<int>(p) = fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
void GMMBuilder::learnGMMs()
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    cv::Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD_MASK )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else if(mask.at<uchar>(p) == GC_FGD_MASK)
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}