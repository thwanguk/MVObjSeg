

#ifndef __CDEMOSUPERPIXEL__
#define __CSUPERPIXEL__

//#include <windows.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "SLIC.h"
//#include "PictureHandler.h"

using namespace cv;

/**
 Superpixel generator --- SLIC
*/


class CSuperPixel
{
public:
	Mat spClMat;  //!< Superpixel colour map
	Mat spPixCnt; //!< Pixel count per superpixel
	int mLabelCnt;//!< Superpixel count
	string imgf;  //!< Source image file name

	CSuperPixel( Mat & , int = 500, int = 20 );
	~CSuperPixel();
	Mat getSPContour();
	Mat getSPMap();
	void getPicture(UINT*&,int&,int&);

private:

	Mat mImg;                 //!< Source image in OpenCV cv::Mat format
	//PictureHandler picHand;   //!< Picture handler
	UINT* pimg;               //!< Source image
	int mWidth;               //!< Width of image
	int mHeight;              //!< Height of image
	int mSpcount;             //!< Superpixel count specified
	int mCompactness;         //!< Superpixel compactness specified
	
	Mat mLabelMapMat;         //!< Superpixel label map in OpenCV cv::Mat format
	Mat mSPContour;           //!< Superpixel contour map
	int* mLabelMap;           //!< Superpixel label map 
	SLIC slic;                //!< SLIC instance

	void run();
	void mGenerateSP();
	void mComputeAverageColour();
	void mDrawContour();
	

};




#endif