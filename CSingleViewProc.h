

#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"

#include "gmmbuilder.h"
#include "multiViewData.h"
#include <vector>

class CSingleViewProc
{
public:
	CSingleViewProc(Mat&, int, int, int, int, int, struct multiViewData*, bool=false, string="");
	CSingleViewProc(Mat&, int, int, int, int, int, int, struct multiViewData*, bool=false, string="");
	~CSingleViewProc(void);
	void downSampling();
	void keyFrameModeling();
	void forceAspectRatio();

	struct multiViewData* m_mvd;

private:
	Mat m_img;
	string m_dir;
	int m_nlayers;
	bool m_isKeyframe;
	int m_camID;
	int m_frmID;
	int m_textonVocabSize;
	int m_siftVocabSize;
	int m_width;
	int m_height;
};

