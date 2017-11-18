

#include "CBoWSIFT.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "dirop.h"
#include <vector>
#include <fstream>
#include "Image.h"
#include "ImageFeature.h"
#include "ImageProcessing.h"


using namespace std;
using namespace cv;

typedef cv::Vec<uchar,128> VecSIFT;

CBoWSIFT::CBoWSIFT(Mat& img, int vocabSize):m_vocabSize(vocabSize)
{
	m_img = img.clone();
	extractSIFT();
}
///*
CBoWSIFT::CBoWSIFT(Mat& img, Mat& vocabulary, int vocabSize):m_vocabSize(vocabSize)
{
	m_img = img.clone();
	m_vocabulary = vocabulary.clone();
	extractSIFT();
}//*/

void CBoWSIFT::extractSIFT()
{
	Mat imMatD;

	m_img.convertTo(imMatD,CV_64FC3);
	DImage im;
	im.loadCVImage(imMatD);
	//bool IsMultiScale = true;
	int cellSize[] = {3};
	vector<int> cellSizeVect(cellSize,cellSize+sizeof(cellSize)/sizeof(int));
	int stepSize = 1;
	bool IsBoundaryIncluded = true;
		
	UCImage imsift;
	ImageFeature::imSIFT(im,imsift,cellSizeVect,stepSize,IsBoundaryIncluded);

	Mat siftMat = Mat(imsift.height(),imsift.width(),CV_8UC(imsift.nchannels()),(uchar*)imsift.pData).clone();
	m_siftMat = Mat(siftMat.rows, siftMat.cols, CV_8UC(siftMat.channels()));
	siftMat.convertTo(m_siftMat, CV_8UC(siftMat.channels()));

	// sift features container
	Mat imDescFull(m_img.cols*m_img.rows,m_siftMat.channels(),CV_8UC1);

	for(int i=0; i<m_siftMat.rows; i++)
	{
		for(int j=0; j<m_siftMat.cols; j++)
		{
			
			VecSIFT desc = m_siftMat.at<VecSIFT>(i,j);
			int k = i*m_siftMat.cols+j;

			for(int l=0; l<m_siftMat.channels(); l++)
				imDescFull.at<uchar>(k,l) = desc[l];			
		}
	}

	imDescFull.convertTo(m_imDesc32,CV_32FC1);
}

CBoWSIFT::~CBoWSIFT(void)
{
}

void CBoWSIFT::buildVocab(Mat& vocabulary, Mat& codeMap, string& dir)
{

	// try loading vocabulary 
	string filename = makeVocabularyFileName(dir+"SIFT", m_vocabSize);
	string mapfilename = makeVocabularyFileName(dir+"SIFT_map", m_vocabSize);

    if( !readVocabulary( filename, m_vocabulary) )
	{
		RNG& rng = theRNG();
		TermCriteria terminate_criterion( CV_TERMCRIT_ITER, 10, 0.0 );

		// initialize trainer
		BOWKMeansTrainer bowTrainer( m_vocabSize, terminate_criterion, 3, KMEANS_PP_CENTERS );

		srand ( time(NULL) );
		int descCnt = 0;

		for(int i=0; i<m_siftMat.rows; i++)
		{
			for(int j=0; j<m_siftMat.cols; j++)
			{
				if((rand()/((double)RAND_MAX + 1.0))<1.00)
				{
					VecSIFT desc = m_siftMat.at<VecSIFT>(i,j);
					Mat imDescVec(1,m_siftMat.channels(),CV_8UC1);
					for(int l=0; l<m_siftMat.channels(); l++)
						imDescVec.at<uchar>(l) = desc[l];
				
					Mat imDescVec32;
					imDescVec.convertTo(imDescVec32,CV_32FC1);
					bowTrainer.add( imDescVec32 );	
					descCnt++;
				}
			}
		}

		//cout <<  ((double)descCnt/(double)m_imDesc32.rows)*100 <<"% features selected for training" << endl; 

		//cout << "Training vocabulary..." << endl;
		//Mat vocabulary;
		m_vocabulary = bowTrainer.cluster();

		if( !writeVocabulary(filename, m_vocabulary) )
		{
			cout << "Error: file " << filename << " can not be opened to write" << endl;
			exit(-1);
		}
	}

	vocabulary = m_vocabulary.clone();
	codeMap = Mat(m_img.rows,m_img.cols,CV_32FC1);

	ifstream iop;
	iop.open(mapfilename);
	if(iop.is_open())
	{
		for(int r=0; r<codeMap.rows; r++)
			for(int c=0; c<codeMap.cols; c++)
				iop >> codeMap.at<float>(r,c);
		iop.close();
	}
	else
	{
		iop.close();
		for(int i=0; i<m_imDesc32.rows; i++)
		{
			float* pdesc = m_imDesc32.ptr<float>(i);
			double minDescDist = 10000000;
			double minVocId;
			Mat descMat(1, m_siftMat.channels(), CV_32F, pdesc);
			for(int j=0; j<m_vocabulary.rows; j++)
			{
				float* pvoc = m_vocabulary.ptr<float>(j);
				Mat vocMat(1, m_siftMat.channels(), CV_32F, pvoc);
				double descDist = 0.0;

				/*
				Mat descDistMat;
				absdiff(descMat, descDist, descDistMat);
				Scalar descDistS = sum(descDistMat);
				descDist += descDistS.val[0];//*/
				///*
				for(int k=0; k<m_siftMat.channels(); k++)
				{
					 descDist += abs(pdesc[k]-pvoc[k]);
				}//*/
				if(descDist<minDescDist)
				{
					minVocId = j;
					minDescDist = descDist;
				}
			}
			int cur_row = static_cast<int>(i/m_img.cols);
			int cur_col = i-cur_row*m_img.cols;
			codeMap.at<float>(cur_row,cur_col) = minVocId;
		}

		ofstream fop;
		fop.open(mapfilename);
		for(int r=0; r<codeMap.rows; r++)
		{
			for(int c=0; c<codeMap.cols; c++)
			{
				fop << codeMap.at<float>(r,c);
				fop << " ";
			}
			fop << endl;
		}
		fop.close();
	}
	

	///*

	/*
	m_CodeWordColorMap = Mat(m_img.rows,m_img.cols,CV_8UC3);
	Vector<Scalar> codeWordColor(m_vocabSize, Scalar::all(-1));

	for(int i=0; i<m_img.rows; i++)
	{
		for(int j=0; j<m_img.cols; j++)
		{
			if(codeWordColor[codeMap.at<float>(i,j)]==Scalar::all(-1))
			{
				RNG& rng = theRNG();
				Scalar color(rng(256), rng(256), rng(256));
				codeWordColor[codeMap.at<float>(i,j)] = color;
			}
			m_CodeWordColorMap.at<Vec3b>(i,j)[0] = codeWordColor[codeMap.at<float>(i,j)].val[0];
			m_CodeWordColorMap.at<Vec3b>(i,j)[1] = codeWordColor[codeMap.at<float>(i,j)].val[1];
			m_CodeWordColorMap.at<Vec3b>(i,j)[2] = codeWordColor[codeMap.at<float>(i,j)].val[2];
		}
	}
	*/
	//imshow("m_CodeWordColorMap", m_CodeWordColorMap);
	//waitKey();
}

void CBoWSIFT::extractCodeMap(Mat& codeMap)
{
	codeMap = Mat(m_img.rows,m_img.cols,CV_32FC1);

	for(int i=0; i<m_imDesc32.rows; i++)
	{
		float* pdesc = m_imDesc32.ptr<float>(i);
		double minDescDist = 10000000;
		double minVocId;
		for(int j=0; j<m_vocabulary.rows; j++)
		{
			float* pvoc = m_vocabulary.ptr<float>(j);
			double descDist = 0.0;
			for(int k=0; k<m_siftMat.channels(); k++)
			{
				 descDist += abs(pdesc[k]-pvoc[k]);
			}
			if(descDist<minDescDist)
			{
				minVocId = j;
				minDescDist = descDist;
			}
		}
		int cur_row = static_cast<int>(i/m_img.cols);
		int cur_col = i-cur_row*m_img.cols;
		codeMap.at<float>(cur_row,cur_col) = minVocId;
		//cout << "minVocId=" << minVocId << endl;
	}
}