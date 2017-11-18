

#ifndef __GMMBUILDER__
#define __GMMBUILDER__

#include "gmm.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

//#include "precomp.hpp"
#include <limits>

using namespace cv;

class GMMBuilder
{

public:

	enum { GC_BGD_MASK    = 2,  //!< background
       GC_FGD_MASK    = 1,  //!< foreground
     };

	Mat img, mask;
	GMMBuilder(const Mat& _img, const Mat& _mask);
	GMM bgdGMM, fgdGMM;

private:

	void initGMMs();
	void assignGMMsComponents();
	void learnGMMs();

	Mat compIdxs;

};


#endif