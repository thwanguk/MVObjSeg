

#include "CSingleViewProc.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"

#include "CTexton.h"
#include "CBoWSIFT.h"
#include "gmmbuilder.h"
#include "grabcutapp.h"
#include "gcgraph.hpp"
#include "graphcutseg.hpp"
#include <vector>

CSingleViewProc::CSingleViewProc(Mat& img, int frmID, int camID, int nlayers,  
								 int textonVocabSize, int siftVocabSize, 
								 struct multiViewData* mvd, bool isKeyframe, string dir):
								 m_nlayers(nlayers),m_textonVocabSize(textonVocabSize),
								 m_siftVocabSize(siftVocabSize),m_isKeyframe(isKeyframe),
								 m_camID(camID),m_dir(dir),m_frmID(frmID)
{
	m_img = img.clone();
	m_mvd = mvd;

	downSampling();
	if(m_isKeyframe)
		keyFrameModeling();
}

CSingleViewProc::CSingleViewProc(Mat& img, int frmID, int camID, int width,  int height,
								 int textonVocabSize, int siftVocabSize, 
								 struct multiViewData* mvd, bool isKeyframe, string dir):
								 m_width(width),m_height(height),m_textonVocabSize(textonVocabSize),
								 m_siftVocabSize(siftVocabSize),m_isKeyframe(isKeyframe),
								 m_camID(camID),m_dir(dir),m_frmID(frmID)
{
	m_img = img.clone();
	m_mvd = mvd;

	forceAspectRatio();
	if(m_isKeyframe)
		keyFrameModeling();
}

CSingleViewProc::~CSingleViewProc(void)
{
}

void CSingleViewProc::forceAspectRatio()
{
				
	//vector<Mat> imgGVec, dsImgGVec, dsImgVec, dsImg64Vec, img64Vec, imgVec;
	Mat img64, imgG;

	cvtColor( m_img, imgG, CV_BGR2GRAY );
	//m_mvd->imgGVec.push_back(imgG.clone());
		
	m_img.convertTo(img64, CV_64FC3);
	m_mvd->img64Vec.push_back(img64.clone());


	Mat dsImg64(cv::Size(m_width, m_height), img64.type());
	resize(img64, dsImg64, cv::Size(m_width, m_height), 0, 0, INTER_CUBIC );
	m_mvd->dsImg64Vec.push_back(dsImg64.clone());
	/*
	Mat dsImg(cv::Size(m_width, m_height), m_img.type());
	resize(m_img, dsImg, cv::Size(m_width, m_height), 0, 0, INTER_CUBIC );
	m_mvd->dsImgVec.push_back(dsImg.clone());//*/

	//Mat dsImgG(cv::Size(m_width, m_height), img64.type());
	//resize(imgG, dsImgG, cv::Size(m_width, m_height), 0, 0, INTER_CUBIC );
	//m_mvd->dsImgGVec.push_back(dsImgG.clone());


}

void CSingleViewProc::downSampling()
{
				
	//vector<Mat> imgGVec, dsImgGVec, dsImgVec, dsImg64Vec, img64Vec, imgVec;
	Mat img64, imgG;

	cvtColor( m_img, imgG, CV_BGR2GRAY );
	//m_mvd->imgGVec.push_back(imgG.clone());
		
	m_img.convertTo(img64, CV_64FC3);
	m_mvd->img64Vec.push_back(img64.clone());

	if(m_nlayers>0)
	{
		Mat dsImg64(cv::Size(img64.cols/pow((double) 2,m_nlayers), img64.rows/pow((double) 2,m_nlayers)), img64.type());
		resize(img64, dsImg64, cv::Size(img64.cols/pow((double) 2,m_nlayers), img64.rows/pow((double) 2,m_nlayers)), 0, 0, INTER_CUBIC );
		m_mvd->dsImg64Vec.push_back(dsImg64.clone());
		/*
		Mat dsImg(cv::Size(m_img.cols/pow((double) 2,m_nlayers), m_img.rows/pow((double) 2,m_nlayers)), m_img.type());
		resize(m_img, dsImg, cv::Size(m_img.cols/pow((double) 2,m_nlayers), m_img.rows/pow((double) 2,m_nlayers)), 0, 0, INTER_CUBIC );
		m_mvd->dsImgVec.push_back(dsImg.clone());//*/

		//Mat dsImgG(cv::Size(img64.cols/pow((double) 2,m_nlayers), img64.rows/pow((double) 2,m_nlayers)), img64.type());
		//resize(imgG, dsImgG, cv::Size(img64.cols/pow((double) 2,m_nlayers), img64.rows/pow((double) 2,m_nlayers)), 0, 0, INTER_CUBIC );
		//m_mvd->dsImgGVec.push_back(dsImgG.clone());
	}
	else
	{
		//m_mvd->dsImgVec.push_back(m_img.clone());
		//imgG32Vec.push_back(imgG32);
		m_mvd->dsImg64Vec.push_back(img64.clone());
		//m_mvd->dsImgGVec.push_back(imgG.clone());
	}
}

void CSingleViewProc::keyFrameModeling()
{
	double t; 

	stringstream cvt;
	cvt << m_camID;
	string strID = cvt.str();

	stringstream cvt2;
	cvt2 << m_frmID;
	string strID2 = cvt2.str();

	string refImgName = m_dir+"keyfrm_f"+strID2+"c"+strID+".png";
	imwrite(refImgName, m_img);
	//imshow("Keyframe", img);
	cout << "  Segment the key frame to extract colour model" << endl;
				
	//Mat mask_pre = imread(refImgName+"_seg_gbc.png" , CV_LOAD_IMAGE_GRAYSCALE);
	string segname = m_dir+"sseg_f" + strID2 + "c" + strID + ".png";
	Mat mask_pre = imread(segname , CV_LOAD_IMAGE_GRAYSCALE);
	Mat segMask;

	string gcsegname = m_dir+"keyfrm_f" + strID2 + "c" + strID + ".png_seg_gbc.png";
	Mat mask_gc = imread(gcsegname , CV_LOAD_IMAGE_GRAYSCALE);
	if( !mask_pre.data && !mask_gc.data)
	{
		grabcutapp( refImgName,  segMask);
		Mat dsSegMask;
		resize(segMask, dsSegMask, cv::Size(m_mvd->dsImg64Vec[0].cols, 
		m_mvd->dsImg64Vec[0].rows), 0, 0, INTER_NEAREST );

		segMask = dsSegMask.clone();
		imwrite(segname, segMask*255);
	}
	else
	{
		if(mask_pre.data)
			segMask = mask_pre.clone()/255;
		else
			segMask = mask_gc.clone();

		if(segMask.cols!=m_mvd->dsImg64Vec[0].cols || segMask.rows!=m_mvd->dsImg64Vec[0].rows)
		{
			Mat dsSegMask;
			resize(segMask, dsSegMask, cv::Size(m_mvd->dsImg64Vec[0].cols, 
			m_mvd->dsImg64Vec[0].rows), 0, 0, INTER_NEAREST );
			segMask = dsSegMask.clone();

			if( !mask_pre.data)
				imwrite(segname, segMask*255);
		}
	}

	Mat img8;
	m_mvd->dsImg64Vec[m_camID].convertTo(img8,CV_8UC3);

	//m_mvd->segMask.push_back(segMask);

	//imshow("segMask", segMask*255);
	//waitKey(0);
	t = (double)cv::getTickCount();
	// build gmm color model based on source image 
	GMMBuilder gmm(img8, 2-segMask); // 2-> BG; 1->FG, gmmbuilder takes uchar
	t = (double)cv::getTickCount() - t;
	cout << "   " << t/cv::getTickFrequency() << " seconds on building GMM colour model" << endl;
				
	//Mat dsSegMask;
	//resize(segMask, dsSegMask, cv::Size(m_mvd->dsImg64Vec[0].cols, 
	//	m_mvd->dsImg64Vec[0].rows), 0, 0, INTER_NEAREST );
	m_mvd->dsSegMask.push_back(segMask);
	
	m_mvd->priorMaskVec.push_back(2-m_mvd->dsSegMask[m_camID].clone()); // downsampled size
	//Mat pMask = Mat::zeros(m_mvd->dsSegMask.rows, m_mvd->dsSegMask.cols, CV_8UC1);
				
	//m_mvd->priorMaskVec.push_back(pMask.clone()); // blanks for the other views

	m_mvd->gmmVec.push_back(gmm);         // same gmm for all views on the first frames

	cout << "  Building texton vocabulary" << endl;
	t = (double)cv::getTickCount();
	// extract texton features from downsampled image
	
	CTexton* textonFactory = new CTexton(img8, m_textonVocabSize);
	Mat textonVocabulary, keyFrameTextonMap;
	textonFactory->buildVocab(textonVocabulary, keyFrameTextonMap,m_dir+"c"+strID);
	m_mvd->textonVocabulary.push_back(textonVocabulary);
	m_mvd->keyFrameTextonMap.push_back(keyFrameTextonMap);
	delete textonFactory;

	t = (double)cv::getTickCount() - t;
	cout << "   " << t/cv::getTickFrequency() << " seconds on building texton vocabulary" << endl;

	cout << "  Building SIFT BoW vocabulary" << endl;
	t = (double)cv::getTickCount();

	// extract dense sift features
	CBoWSIFT* BoWFactory = new CBoWSIFT(img8, m_siftVocabSize);
	Mat siftVocabulary, keyFrameSIFTCodeMap;
	BoWFactory->buildVocab(siftVocabulary, keyFrameSIFTCodeMap,m_dir+"c"+strID);
	m_mvd->siftVocabulary.push_back(siftVocabulary);
	m_mvd->keyFrameSIFTCodeMap.push_back(keyFrameSIFTCodeMap);
	delete BoWFactory;

	t = (double)cv::getTickCount() - t;
	cout << "   " << t/cv::getTickFrequency() << " seconds on building SIFT BoW vocabulary" << endl;

	t = (double)cv::getTickCount();
	///*
	/// Set the ranges of histogram
	int histSize = m_textonVocabSize;
	float range[] = { 0, m_textonVocabSize } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat f_hist, b_hist;

	/// Compute the FG histograms:
	calcHist( &m_mvd->keyFrameTextonMap[m_camID], 1, 0, m_mvd->dsSegMask[m_camID], f_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &m_mvd->keyFrameTextonMap[m_camID], 1, 0, 1-m_mvd->dsSegMask[m_camID], b_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histogram
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImageFG( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	Mat histImageBG( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	f_hist = f_hist/(sum(f_hist).val[0]);
	b_hist = b_hist/(sum(b_hist).val[0]);

	m_mvd->textonHist.push_back(b_hist);
	m_mvd->textonHist.push_back(f_hist);

	t = (double)cv::getTickCount() - t;
	cout << "   " << t/cv::getTickFrequency() << " seconds on building Texton BG/FG histograms" << endl;

	//normalize(f_hist, f_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw histogram
	/*
	for( int i = 1; i < histSize; i++ )
	{
		line( histImageFG, cv::Point( bin_w*(i-1), hist_h - cvRound(f_hist.at<float>(i-1)*histImageFG.rows) ) ,
						cv::Point( bin_w*(i), hist_h - cvRound(f_hist.at<float>(i)*histImageFG.rows) ),
						Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImageBG, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)*histImageBG.rows) ) ,
						cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)*histImageBG.rows) ),
						Scalar( 0, 0, 255), 2, 8, 0  );
	}

	/// Display
	imshow("FG texton hist", histImageFG );
	imshow("BG texton hist", histImageBG );
	waitKey();//*/
}