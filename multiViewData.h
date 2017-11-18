

#ifndef __MULTIVIEWDATA__
#define __MULTIVIEWDATA__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "gmmbuilder.h"
#include <deque>

using namespace cv;
using namespace std;

struct multiViewData
{
	vector<vector<Mat> > textonLikelihoodVec;
	vector<GMMBuilder> gmmVec;
	vector<Mat> dsImg64Vec;
	vector<Mat> preDsImg64Vec;
	vector<Mat> priorMaskVec;
	vector<Mat> textonMapVec;
	vector<Mat> dsSpVec;
	//vector<Mat> spVec;
	vector<int> lCntVec;
	vector<Mat> spClVec;
	vector<vector<vector<float> > > SPProbVec;
	vector<vector<vector<Mat> > > histVec;
	vector<vector<cv::Point2f> > centDsSPViewVec;
	vector<Mat> flowVec;
	vector<Mat> preMaskVec;
	vector<Mat> fundamMatVec;

	//vector<Mat> imgGVec;
	//vector<Mat> dsImgGVec;
	//vector<Mat> dsImgVec;
	vector<Mat> img64Vec; 
	//vector<Mat> imgVec;

	//vector<Mat> segMask;
	vector<Mat> dsSegMask;
	//Mat textonVocabulary, keyFrameTextonMap;
	vector<Mat> textonVocabulary, keyFrameTextonMap;
	vector<Mat> siftVocabulary, keyFrameSIFTCodeMap;
	//Mat siftVocabulary, keyFrameSIFTCodeMap;
	vector<Mat> textonHist;
	//vector<Mat> SPMaskVec;
	vector<Mat> siftCodeMapVec;
	vector<Mat> disparityMapVec;
	vector<Mat> textonProbVec;
	vector<vector<int> > spDispVec;
	int tolLabel;
};

#endif