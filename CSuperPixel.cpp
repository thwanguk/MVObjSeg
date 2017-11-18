

#include <windows.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "SLIC.h"
//#include "PictureHandler.h"
#include "CSuperPixel.h"
using namespace cv;

//! Constructor
/*!
    \param imgFile the image file name.
*/
/*
CSuperPixel::CSuperPixel( Mat & img8 ):mSpcount(500),mCompactness(20) 
{
	mImg = img8.clone();
	run();
}
//! Constructor
/*!
    \param imgFile the image file name.
	\param spCnt the specified superpixel count
	\param spCompact the specified superpixel compactness
*/
CSuperPixel::CSuperPixel( Mat & img8, int spCnt, int spCompact )
{
	//imgf = imgFile;
	mImg = img8.clone();
	mSpcount = spCnt;
	mCompactness = spCompact;
	run();
}

//! Destructor
CSuperPixel::~CSuperPixel()
{
	if(mLabelMap) delete [] mLabelMap;
	if(pimg) delete [] pimg;
}

//! Member function that calls SLIC to produce superpixels, compute average superpixel colour and draw contours
void CSuperPixel::run()
{
	//picHand.GetPictureBuffer( imgf, pimg, mWidth, mHeight );
	getPicture( pimg, mWidth, mHeight );
	mLabelMap = new int[mWidth*mHeight];
	//mImg = imread(imgf);

	mGenerateSP();
	mComputeAverageColour();
	mDrawContour();
}

void CSuperPixel::getPicture( UINT*& imgBuffer, int& width, int& height )
{
	//Mat img8 = imread(filename);
	width = mImg.cols;
	height = mImg.rows;

	Mat imgRGB;
	cv::cvtColor(mImg,imgRGB,CV_BGR2RGB);

	vector<Mat> RGB_planes;
	split(imgRGB, RGB_planes);
	vector<Mat> ARGB_planes;
	ARGB_planes.push_back(Mat::zeros(height,width,CV_8UC1));
	ARGB_planes.push_back(RGB_planes[0]);
	ARGB_planes.push_back(RGB_planes[1]);
	ARGB_planes.push_back(RGB_planes[2]);

	Mat imgARGB;
	cv::merge(ARGB_planes,imgARGB);

	int imgSize = width*height;
	imgBuffer = new UINT[imgSize];
	memcpy( imgBuffer, (UINT*)imgARGB.data, imgSize*sizeof(UINT) );

}

//! Member function that calls SLIC to produce superpixels
void CSuperPixel::mGenerateSP()
{
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels( pimg, mWidth, mHeight, mLabelMap, mLabelCnt, mSpcount, mCompactness );
	mLabelMapMat = Mat( mHeight, mWidth, CV_32FC1 );
	cv::Point p;
	for(p.y=0;p.y<mHeight;p.y++)
		for(p.x=0;p.x<mWidth;p.x++)
			mLabelMapMat.at<float>(p.y,p.x) =  saturate_cast<float>(mLabelMap[p.x+p.y*mWidth]);
}

//! Member function that draws superpixel contours
void CSuperPixel::mDrawContour()
{
	slic.DrawContoursAroundSegments( pimg, mLabelMap, mWidth, mHeight, 0 );

	Mat imgARGB(mHeight,mWidth,CV_8UC4, (uchar*)pimg);
	vector<Mat> RGB_planes, ARGB_planes;
	split(imgARGB, ARGB_planes);

	vector<Mat> BGR_planes;
	BGR_planes.push_back(ARGB_planes[3]);
	BGR_planes.push_back(ARGB_planes[2]);
	BGR_planes.push_back(ARGB_planes[1]);

	cv::merge(BGR_planes,mSPContour);
	//imwrite("./tempframe_SLIC.jpg", mSPContour);


}

//! Member function that returns superpixel contours
/*!
	\return superpixel contour map
*/
Mat CSuperPixel::getSPContour()
{
	return mSPContour.clone();
}

//! Member function that returns superpixel map
/*!
	\return superpixel map
*/
Mat CSuperPixel::getSPMap()
{
	return mLabelMapMat.clone();
}

//! Member function that computes average superpixel colour
void CSuperPixel::mComputeAverageColour()
{
	spClMat  = Mat::zeros(1, mLabelCnt, CV_64FC3);
	spPixCnt = Mat::zeros(1, mLabelCnt, CV_64FC3);
	Mat img64;
	mImg.convertTo(img64, CV_64FC3);

	cv::Point p;
	for(p.y=0;p.y<mHeight;p.y++)
	{
		for(p.x=0;p.x<mWidth;p.x++)
		{
			int labelID = (int)mLabelMapMat.at<float>(p.y,p.x);
			spPixCnt.at<double>(0,labelID) += 1;
			spClMat.at<Vec3d>(0,labelID)[0] += img64.at<Vec3d>(p.y,p.x)[0];
			spClMat.at<Vec3d>(0,labelID)[1] += img64.at<Vec3d>(p.y,p.x)[1];
			spClMat.at<Vec3d>(0,labelID)[2] += img64.at<Vec3d>(p.y,p.x)[2];
		}
	}
	
	for(int i=0; i<mLabelCnt; i++)
	{
		spClMat.at<Vec3d>(0,i)[0] = spClMat.at<Vec3d>(0,i)[0] / spPixCnt.at<double>(0,i); 
		spClMat.at<Vec3d>(0,i)[1] = spClMat.at<Vec3d>(0,i)[1] / spPixCnt.at<double>(0,i); 
		spClMat.at<Vec3d>(0,i)[2] = spClMat.at<Vec3d>(0,i)[2] / spPixCnt.at<double>(0,i); 
	}
}