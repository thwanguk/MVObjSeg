#include <windows.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "mrfstereo.h"
#include "drawFlow.h"
#include "gmmbuilder.h"
#include "SLIC.h"

#include "grabcutapp.h"
#include "gcgraph.hpp"
#include "graphcutseg.hpp"

#include "CInterViewGraph.h"
#include "CSuperPixel.h"
#include "CBandedGraphCut.h"
#include "CTexton.h"
#include "CBoWSIFT.h"
#include "multiViewData.h"
#include "CSingleViewProc.h"
#include "ui.h"

#include "svm.h"
#include <deque>
#include <fstream>

#include "Image.h"
#include "BPFlow.h"
#include "stdio.h"
#include "time.h"
#include "ImageFeature.h"
#include "ImageProcessing.h"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */

const int textonVocabSize = 400;
const int siftVocabSize = 200;
const int m_spcount = 3000;// #superpixels
const int m_compactness = 20; //1-80

GCUI gcui;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct persist_model {
    struct svm_model * mod;     // svmlibs model
    struct svm_node * x_space;  // field of sparse training vectors
};

void genSP(struct multiViewData * mvd, int i, int frameID);
void dsSPMaskCent(int i, int l, struct multiViewData * mvd);
void extractTexton(int i, struct multiViewData * mvd, string dir);
void calTextonLikelihood(struct multiViewData * train_mvd, struct multiViewData * mvd, int i, vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec);
void calTextonLikelihood(vector<Mat>& textonHist, struct multiViewData * mvd, int i, vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec);
void calAllTextonLikelihood(vector<Mat>& textonHist, struct multiViewData * mvd, int i, vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec);
void extractSIFTBoW(int i, struct multiViewData * mvd, string dir);
void svmTrainingData(int i, struct multiViewData * mvd, struct svm_problem& prob, struct svm_node * x_space);//,
					//vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec);

void svmTestingData(int i, int j, struct multiViewData * mvd, struct svm_node *temp_vec, 
					vector<Mat>& textonHistVec,
			vector<Mat>& labHistVec, vector<Mat>& siftHistVec);
void matchingToFundamMat(struct multiViewData * mvd);
void calibToFundamMat(struct multiViewData * mvd, string& calibfile, string frmidx);
void loadCameraParam(string& calibfile, vector<Mat>& intrinsicVec, vector<Mat>& distVec,
					 vector<Mat>& RVec, vector<Mat>& TVec);
void stereoMatchingMRF(Mat& l_rect_ds, Mat& r_rect_ds, Mat& dispMat);
void calFundamMat(Mat& M1, Mat& M2, Mat& R, Mat& T, Mat& F);
void rectifyStereoImage(Mat& imgL, Mat& imgR, Mat& M1, Mat& M2, Mat& R, Mat& T, Mat& D1, 
						Mat& D2, Mat& l_rect, Mat& r_rect, Mat& H1, Mat& H2, Mat& tML, Mat& tMR, Mat& Q);

void svmTraining(struct persist_model *rv, struct svm_parameter & param, int i, struct multiViewData * mvd,
				 //vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW);//, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec);

void svmClassification(struct persist_model *rv, int i, Mat& fgProbMap, Mat& bgProbMap, struct multiViewData * mvd,
					   vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec, string dir);
void svmAllClassification(vector<struct persist_model *> rvec, int i, Mat& fgProbMap, Mat& bgProbMap, struct multiViewData * mvd,
					   vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec, string dir);
void calcSIFTFlow(Mat& Im1, Mat& Im2, Mat& flow);

void spDisparityMap(struct multiViewData * mvd);
void on_mouseclick( int event, int x, int y, int flags, void* param );

//string dir = "./Queen/";
//string dir = "./karate/";
//string dir = "./fallingdown/";
//string dir = "./dance/";
//string dir = "./salsa/";

int main( int argc, char** argv )
{

	string vidTrainList[] = {"000/00000.png", "001/00000.png","002/00000.png",
		"003/00000.png","004/00000.png","005/00000.png","006/00000.png","007/00000.png"}; //karate

	int startFrm = 0;

	int nView = 8;
	vector<string> vidTrainListVec;
	vidTrainListVec.assign(vidTrainList, vidTrainList+nView);

	string vidList[] = {"000/","001/","002/",
		"003/","004/","005/","006/","007/"}; //karate
	
	string calibFile =  dir+"calibration.txt";

	int gcLevels = 3;

	struct svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;//100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	vector<struct persist_model*> rv; // model learned from full feature
	//struct persist_model* rv;
	

	int histSizeLab = 23;
	int histSize = textonVocabSize;
	int histSizeBoW = siftVocabSize;

	int wl[] = { 1, -1 };
	double w[] = { 1.0 , 1.0 };
	int l = gcLevels-1;

	struct multiViewData* train_mvd = new struct multiViewData;
	train_mvd->tolLabel = 0;

	string sfrmidx;
	stringstream sfstrconvt;
	sfstrconvt << startFrm;
	sfrmidx = sfstrconvt.str();

	cout << "- Model the keyframe" << endl;
	double tk = getTickCount(); // timer
	for(int i=0; i<vidTrainListVec.size(); i++)
	{

		string camID;          // string which will contain the result
		stringstream convert;   // stream used for the conversion
		convert << i;      // insert the textual representation of 'k' in the characters in the stream
		camID = convert.str(); // set 'idstr' to the contents of the stream

		CSingleViewProc* SVP;

		Mat trainImg = imread(dir+vidTrainListVec[i]);
		SVP = new CSingleViewProc(trainImg, startFrm, i, l, textonVocabSize, siftVocabSize, train_mvd, true, dir);
		delete SVP;

		genSP(train_mvd, i, 0);
		dsSPMaskCent(i, l, train_mvd);
		extractTexton(i, train_mvd, "f"+sfrmidx+"c"+camID);
		/*
		vector<Mat> textonLikelihoodFGVec;
		vector<Mat> textonLikelihoodBGVec;
		calTextonLikelihood(train_mvd, train_mvd, i, textonLikelihoodFGVec,textonLikelihoodBGVec);*/

		extractSIFTBoW(i, train_mvd, "f"+sfrmidx+"c"+camID);

		double t = (double)cv::getTickCount();

		struct persist_model* svmModel = Malloc(struct persist_model, 1);

		svmTraining(svmModel, param, i, train_mvd, histSizeLab, histSizeBoW);

		rv.push_back(svmModel);

		t = (double)cv::getTickCount() - t;
		cout << "    " << t/cv::getTickFrequency() << " seconds on learning SVM" << endl;
	}

	/*
	double t = (double)cv::getTickCount();

	rv = Malloc(struct persist_model, 1);

	svmAllTraining(rv, param, train_mvd, histSizeLab, histSizeBoW);


	t = (double)cv::getTickCount() - t;
	cout << "    " << t/cv::getTickFrequency() << " seconds on learning SVM" << endl;*/

	tk = (double)cv::getTickCount() - tk;
	cout << " " << tk/cv::getTickFrequency() << " seconds on keyframe modeling" << endl;

	vector<Mat> textonVocabulary = train_mvd->siftVocabulary;
	vector<Mat> siftVocabulary = train_mvd->siftVocabulary;
	vector<GMMBuilder> gmmVec = train_mvd->gmmVec;//*/
	vector<Mat> textonHist = train_mvd->textonHist;

	vector<Mat> preDsImg64Vec = train_mvd->dsImg64Vec;
	vector<Mat> preMaskVec = train_mvd->priorMaskVec;

	//imshow("preMaskVec0",(2-preMaskVec[0])*255); waitKey();

	delete train_mvd;
	double t;

	// iterate successive frames
	for(int frmCnt=startFrm+1; frmCnt<100; frmCnt++)
	{
		double tt = (double)cv::getTickCount(); 
		
		//string vidListTemp[8];
		string* vidListTemp = new string [nView];

		string frmidx;
		stringstream fstrconvt;
		fstrconvt << frmCnt;
		frmidx = fstrconvt.str();

		char frameName[100];
		sprintf(frameName, "%05d", frmCnt);
		cout << "Segmenting frame #" << frameName << endl;
		string strFrameName(frameName);
		

		for(int v=0; v<nView; v++)
			vidListTemp[v] = vidList[v]+strFrameName+".png";

		vector<string> vidListVec;
		vidListVec.assign(vidListTemp, vidListTemp+nView);
	      
		vector<Mat> maskVec;

		struct multiViewData* mvd = new struct multiViewData;
		mvd->tolLabel = 0;


		///////////////// Generate superpixels /////////////////
		cout << "- Generate superpixels" << endl;
		bool segFlag = false;
		// iterate all views 
		for(int i=0; i<vidListVec.size(); i++)
		{
			string camidx;
			stringstream strconvt;
			strconvt << i;
			camidx = strconvt.str();

			Mat img;
			string fileName = dir+vidListTemp[i];
			//cout << "fileName = " << fileName << endl;
			img = imread(fileName);

			if (img.empty())
				break;	

			string segname = dir+"sseg_f" + frmidx + "c" + camidx + ".png";
			Mat segImg = imread(segname,0);
			if(segImg.data)
			{
				//cout << "ok" << endl;
				Mat segImg32, img64, dsImg64, segImg8;
				//segImg.convertTo(segImg32,CV_32FC1);
				//segImg.convertTo(segImg8, CV_8UC1);
				preMaskVec[i] = 2-segImg/255;

				img.convertTo(img64, CV_64FC3);
				if(l>0)
					resize(img64, dsImg64, cv::Size(img64.cols/pow((double) 2,l), img64.rows/pow((double) 2,l)), 0, 0, INTER_CUBIC );
				else
					dsImg64 = img64.clone();

				preDsImg64Vec[i] = dsImg64.clone();
				segFlag = true;

				continue;
			}



			CSingleViewProc* SVP;

			//else
			SVP = new CSingleViewProc(img, frmCnt, i, l, textonVocabSize, siftVocabSize, mvd);

			Mat pMask = Mat::zeros(mvd->dsImg64Vec[0].rows, mvd->dsImg64Vec[0].cols, CV_8UC1);	
			mvd->priorMaskVec.push_back(pMask.clone()); // blanks for the other views
	
			delete SVP;


		}

		if(segFlag) 
		{
			delete mvd;
			continue;
		}
		

		//if((frmCnt>startFrm) && (frmCnt-startFrm)%2==0)
		if(frmCnt>startFrm+1)
		{
			gmmVec.clear();
			for(int i=0; i<vidListVec.size(); i++)
			{
				string camidx;
				stringstream strconvt;
				strconvt << i;
				camidx = strconvt.str();

				string prefrmidx;
				stringstream fstrconvt2;
				fstrconvt2 << frmCnt-1;
				prefrmidx = fstrconvt2.str();

				Mat img;
				Mat img64 = preDsImg64Vec[i];
				img64.convertTo(img, CV_8UC3);
				
				string segname = dir+"sseg_f" + prefrmidx + "c" + camidx + ".png";
				Mat segImg = imread(segname,0);
				GMMBuilder gmm(img, 2-segImg/255);
				gmmVec.push_back(gmm);
			}
		}

		mvd->gmmVec = gmmVec;
		mvd->textonVocabulary = textonVocabulary;
		mvd->siftVocabulary = siftVocabulary;
		mvd->preMaskVec = preMaskVec;	
		mvd->preDsImg64Vec = preDsImg64Vec;

		//imshow("preMaskVec",(2-preMaskVec[0])*255); waitKey();

		//vector<Mat> SPContourVec;
		vector<Mat> textonLikelihoodFGVec;
		vector<Mat> textonLikelihoodBGVec;
		//vector<Mat> siftCodeMapVec;
		vector<vector<Mat> > textonHistViewVec;
		vector<vector<Mat> > siftHistViewVec;
		vector<vector<Mat> > labHistViewVec;
		vector<vector<float> > SPFGProbViewVec;
		vector<vector<float> > SPBGProbViewVec;

		string winName = "UI";
		namedWindow(winName, CV_WINDOW_NORMAL);
		Mat UIImage(mvd->dsImg64Vec[0].rows*2, mvd->dsImg64Vec[0].cols*4, CV_8UC3);

		for( int i=0; i<mvd->img64Vec.size(); i++)
		{
			Mat img8;

			mvd->dsImg64Vec[i].convertTo(img8, CV_8UC3);
			
			Rect r((i%4)*mvd->dsImg64Vec[0].cols,(i/4)*mvd->dsImg64Vec[0].rows,mvd->dsImg64Vec[0].cols,mvd->dsImg64Vec[0].rows);
			Mat dstRoi = UIImage(r);
			img8.convertTo(dstRoi, dstRoi.type(), 1, 0);
		}

		//imshow(winName, UIImage);
		cvSetMouseCallback( winName.c_str(), on_mouseclick, 0 );
		//waitKey();

		for( int i=0; i<mvd->img64Vec.size(); i++)
		{

			string camidx;
			stringstream strconvt;
			strconvt << i;
			camidx = strconvt.str();
	
			genSP(mvd, i, frmCnt);
			dsSPMaskCent(i, l, mvd);

			t = (double)cv::getTickCount();

			// extract texton features from downsampled image
			
			extractTexton(i, mvd, "f"+frmidx+"c"+camidx);
			//calTextonLikelihood(train_mvd, mvd, i, textonLikelihoodFGVec,textonLikelihoodBGVec);
			calAllTextonLikelihood(textonHist, mvd, i, textonLikelihoodFGVec,textonLikelihoodBGVec);

			t = (double)cv::getTickCount() - t;
			cout << "    " << t/cv::getTickFrequency() << " seconds on texton extraction" << endl;


			vector<Mat> textonHistVec;
			vector<Mat> labHistVec;
			vector<Mat> siftHistVec;

			t = (double)cv::getTickCount();
			
			extractSIFTBoW(i, mvd, "f"+frmidx+"c"+camidx);
			t = (double)cv::getTickCount() - t;
			cout << "    " << t/cv::getTickFrequency() << " seconds on SIFT BoW extraction" << endl;


			vector<float> SPFGProbVec(mvd->lCntVec[i],0);
			vector<float> SPBGProbVec(mvd->lCntVec[i],0);


			t = (double)cv::getTickCount();

			Mat fgProbMap(mvd->dsSpVec[i].rows, mvd->dsSpVec[i].cols, CV_32FC1);
			Mat bgProbMap(mvd->dsSpVec[i].rows, mvd->dsSpVec[i].cols, CV_32FC1);

			svmAllClassification(rv, i, fgProbMap, bgProbMap, mvd, textonHistVec, 
				labHistVec, siftHistVec, histSizeLab, histSizeBoW, SPFGProbVec, SPBGProbVec, dir+"f"+frmidx+"c"+camidx);


			imshow("fgProbMap "+camidx, fgProbMap);
			//imshow("bgProbMap "+frmid, bgProbMap);	

			t = (double)cv::getTickCount() - t;
			cout << "    " << t/cv::getTickFrequency() << " seconds on SVM classification" << endl;

			//}

			textonHistViewVec.push_back(textonHistVec);
			siftHistViewVec.push_back(siftHistVec);
			labHistViewVec.push_back(labHistVec);
			SPFGProbViewVec.push_back(SPFGProbVec);
			SPBGProbViewVec.push_back(SPBGProbVec);

		}

		

		mvd->histVec.push_back(textonHistViewVec);
		mvd->histVec.push_back(siftHistViewVec);
		mvd->histVec.push_back(labHistViewVec);
		mvd->textonLikelihoodVec.push_back(textonLikelihoodBGVec);
		mvd->textonLikelihoodVec.push_back(textonLikelihoodFGVec);
		mvd->SPProbVec.push_back(SPBGProbViewVec);
		mvd->SPProbVec.push_back(SPFGProbViewVec);

		textonHistViewVec.clear();
		siftHistViewVec.clear();
		labHistViewVec.clear();
		textonLikelihoodBGVec.clear();
		textonLikelihoodFGVec.clear();
		SPFGProbViewVec.clear();
		SPBGProbViewVec.clear();

		// estimate fundamental matrix via feature matching
		//matchingToFundamMat(mvd);

		t = (double)cv::getTickCount();
		calibToFundamMat(mvd, calibFile, frmidx);
		spDisparityMap(mvd);
		t = (double)cv::getTickCount() - t;
		cout << "  " << t/cv::getTickFrequency() << " seconds on computing Fundamental Matrix" << endl;

		///*
		//vector<Mat> flowVec;

		bool flowFlag = true;
		if(flowFlag)
		{
			for(int i=0; i<mvd->dsImg64Vec.size(); i++)
			{
				t = (double)cv::getTickCount();

				Mat flow;
				
				Mat imgG1, imgG2, img8G1, img8G2;
				preDsImg64Vec[i].convertTo(img8G1,CV_8UC3);
				mvd->dsImg64Vec[i].convertTo(img8G2,CV_8UC3);
				cv::cvtColor(img8G1, imgG1, CV_BGR2GRAY);
				cv::cvtColor(img8G2, imgG2, CV_BGR2GRAY);

				Mat cflow = img8G1.clone();
				//Mat cflow = imgG1.clone();

				//calcSIFTFlow(preDsImg64Vec[i], mvd->dsImg64Vec[i], flow);
				calcOpticalFlowFarneback(imgG1, imgG2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
				//calcOpticalFlowSF(img8G1, img8G2, flow, 3, 2, 15, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);


				mvd->flowVec.push_back(flow.clone());
				if(i==7)
				{
					drawOptFlowMap(flow, cflow, 8, 5, CV_RGB(0, 255, 0));
					imshow("Flow", cflow);
				}

				t = (double)cv::getTickCount() - t;
				cout << "  " << t/cv::getTickFrequency() << " seconds on computing optical flow #" << i << endl;
				//waitKey();

				
			}
		}

		cout << "- Constructing graph" << endl;

		
		t = (double)cv::getTickCount();
		CInterViewGraph* ngragh = new CInterViewGraph(mvd, 20);//2

		cout << "- Iteration #0 finished" << endl;

		vector<Mat> spSegMask;
		ngragh->getMask(maskVec, spSegMask);

		for(int k = 0; k<maskVec.size(); k++)
		{
			string idstr;          // string which will contain the result
			stringstream convert;   // stream used for the conversion
			convert << k;      // insert the textual representation of 'k' in the characters in the stream
			idstr = convert.str(); // set 'idstr' to the contents of the stream

			string winname = "sseg_f" + frmidx + "c" + idstr;
			imshow("sseg_c" + idstr, maskVec[k]*255);
			imwrite(dir+winname+".png",maskVec[k]*255);
			maskVec[k] = 2-maskVec[k];
			
		}

		Mat gPriMask = Mat::zeros(mvd->img64Vec[0].rows*2, mvd->img64Vec[0].cols*4, CV_8UC1);

		gcui.setImageAndWinName( UIImage, winName, gPriMask);
		gcui.showImage();

		int itc = 0;
		bool exitcode = false;
		for(;;)
		{

			int c = cvWaitKey(0);//360000

			if(c==-1) break;

			switch( (char) c )
			{
				case '\x1b':
					cout << "Exiting ..." << endl;
					exitcode = true;
					break;
				case 'r':
					cout << endl;
					gcui.reset();
					gcui.showImage();
					break;

				case 'n':

					//
					delete ngragh;
					for(int i=0; i<mvd->dsImg64Vec.size(); i++)
					{
						Mat priMask;
						Rect r((i%4)*mvd->dsImg64Vec[0].cols,(i/4)*mvd->dsImg64Vec[0].rows,mvd->dsImg64Vec[0].cols,mvd->dsImg64Vec[0].rows);
						gPriMask(r).convertTo(priMask, gPriMask.type());
						mvd->priorMaskVec[i] = priMask.clone();

						string idstr;          // string which will contain the result
						stringstream convert;   // stream used for the conversion
						convert << i;      // insert the textual representation of 'k' in the characters in the stream
						idstr = convert.str(); // set 'idstr' to the contents of the stream

						//namedWindow("gPriMask",CV_WINDOW_NORMAL);
						//imshow("gPriMask "+idstr, 127*(2-mvd->priorMaskVec[i]));// waitKey();
					}

					
					ngragh = new CInterViewGraph(mvd, 20);//2

					cout << "- Iteration #" << ++itc << " finished" << endl;

					ngragh->getMask(maskVec, spSegMask);

					for(int k = 0; k<maskVec.size(); k++)
					{
						string idstr;          // string which will contain the result
						stringstream convert;   // stream used for the conversion
						convert << k;      // insert the textual representation of 'k' in the characters in the stream
						idstr = convert.str(); // set 'idstr' to the contents of the stream

						string winname = "sseg_f" + frmidx + "c" + idstr;
						imshow("sseg_c" + idstr, maskVec[k]*255);
						imwrite(dir+winname+".png",maskVec[k]*255);
						maskVec[k] = 2-maskVec[k];
			
					}
			}

			if(exitcode) break;	
		}
		
		/*
		for(int k = 0; k<maskVec.size(); k++)
		{
			Mat spMaskMap(mvd->dsSpVec[k].rows, mvd->dsSpVec[k].cols, CV_8UC1);
			cv::Point p;
			for(p.y=0; p.y<mvd->dsSpVec[k].rows; p.y++)
			{
				for(p.x=0; p.x<mvd->dsSpVec[k].cols; p.x++)
				{
					spMaskMap.at<uchar>(p) = spSegMask[k].at<uchar>(0,mvd->dsSpVec[k].at<float>(p));
				}
			}

			string idstr;          // string which will contain the result
			stringstream convert;   // stream used for the conversion
			convert << k;      // insert the textual representation of 'k' in the characters in the stream
			idstr = convert.str(); // set 'idstr' to the contents of the stream

			string winname = "SP Labeling Map "+ idstr;

			imshow(winname, spMaskMap*255);

		}*/

		
		//waitKey();

		// retrieve matched superpixels only
		//vector<vector<char> > matchesMaskVec = ngragh->getSPMatchesMask();

		delete ngragh;
		t = (double)cv::getTickCount() - t;
		cout << "  " << t/cv::getTickFrequency() << " seconds on downsampled segmentation" << endl;

		/*//-- Draw correspondence of superpixels
		Mat img_matches;
		drawMatches( SPContourVec[0], keypointsVec[0], SPContourVec[1], keypointsVec[1], matchesVec[0], img_matches,
			Scalar::all(-1), Scalar::all(-1), matchesMaskVec[0], 2 ); 

		//-- Show detected matches
		imshow("Matches", img_matches );*/
		//waitKey();

		//mvd->preMaskVec = maskVec; // should be substracted from 2
		

		/*
		cout << "- Perform multiple layer graphcut" << endl;
		t = (double)cv::getTickCount();

		const double gamma = 10;//10;

		Mat mask;

		for(int k = 0; k<mvd->dsImg64Vec.size(); k++)
		{
			if(l>0)
			{
				CBandedGraphCut* bgraphcut = new CBandedGraphCut(mvd->img64Vec[k],maskVec[k],l,gamma,gmmVec[k]);
				mask = bgraphcut->getMask();
				//maskTopLVec.push_back(mask);
				delete bgraphcut;
			}
			else
			{
				//maskTopLVec = maskVec;
				mask = maskVec[k];
			}

			string idstr;          // string which will contain the result
			stringstream convert;   // stream used for the conversion
			convert << k;      // insert the textual representation of 'k' in the characters in the stream
			idstr = convert.str(); // set 'idstr' to the contents of the stream

			string winname = "mask "+ idstr;
			imshow(winname, mask*255);
			imwrite(winname+".png", mask*255);
			waitKey();
			//maskTopLVec.push_back(mask);
				
		}
		t = (double)cv::getTickCount() - t;
		cout << "  " << t/cv::getTickFrequency() << " seconds on multi-level segmentation" << endl;

		*/
		
		preDsImg64Vec = mvd->dsImg64Vec;
		preMaskVec = maskVec;

		//delete ngragh;
		delete mvd;

		tt = (double)cv::getTickCount() - tt;
		cout << "- Totally " << tt/cv::getTickFrequency() << " seconds on the current frame" << endl;
		//waitKey();
	}
	return 0;

}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./" << std::endl; }

void dsSPMaskCent(int i, int l, struct multiViewData * mvd)
{
	int width = mvd->img64Vec[i].cols;
	int height = mvd->img64Vec[i].rows;

	/*
	// down sampling mask
	if(l>0)
	{
		Mat dsLabelsImg;//(cv::Size(width/pow((double) 2,l), height/pow((double) 2,l)), CV_32FC1);
		resize(mvd->spVec[i], dsLabelsImg, cv::Size(width/pow((double) 2,l), height/pow((double) 2,l)), 0, 0, INTER_NEAREST ); // nearest
		mvd->dsSpVec.push_back(dsLabelsImg.clone());
	}
	else
		mvd->dsSpVec.push_back(mvd->spVec[i].clone());*/

	// per sp-label mask for building texton histogram
	//vector<Mat> SPMaskVec;
	
	vector<cv::Point2f> centPVec(mvd->lCntVec[i],cv::Point2f(0,0));
	vector<int> pixCntVec(mvd->lCntVec[i],0);
	cv::Point p;
	//cout << "centPVec.size() = " << centPVec.size() << endl;

	for(p.y=0; p.y<mvd->dsSpVec[i].rows; p.y++)
	{
		for(p.x=0; p.x<mvd->dsSpVec[i].cols; p.x++)
		{
			//cout << "mvd->dsSpVec[i].at<float>(p) = " << mvd->dsSpVec[i].at<float>(p) << endl;
			centPVec[mvd->dsSpVec[i].at<float>(p)].x += p.x;
			centPVec[mvd->dsSpVec[i].at<float>(p)].y += p.y;
			pixCntVec[mvd->dsSpVec[i].at<float>(p)]++;
		}
	}
	for(int j=0; j<mvd->lCntVec[i]; j++)
	{
		centPVec[j].x /= (float) pixCntVec[j];
		centPVec[j].y /= (float) pixCntVec[j];
	}
	mvd->centDsSPViewVec.push_back(centPVec);
}

void extractSIFTBoW(int i, struct multiViewData * mvd, string fdir)
{
	string idstr;          // string which will contain the result
	stringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'k' in the characters in the stream
	idstr = convert.str(); // set 'idstr' to the contents of the stream
	string BoWFileName = dir+"BoWMap_"+fdir+".txt";

	Mat SIFTCodeMap;
	ifstream iop;
	iop.open(BoWFileName);
	if(iop.is_open())
	{
		SIFTCodeMap = Mat(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		for(int r=0; r<SIFTCodeMap.rows; r++)
			for(int c=0; c<SIFTCodeMap.cols; c++)
				iop >> SIFTCodeMap.at<float>(r,c);
		iop.close();
	}
	else
	{
		iop.close();
		Mat img8;
		mvd->dsImg64Vec[i].convertTo(img8,CV_8UC3);

		CBoWSIFT BoWFactory(img8, mvd->siftVocabulary[i], siftVocabSize);
		BoWFactory.extractCodeMap(SIFTCodeMap);

		ofstream fop;
		fop.open(BoWFileName);
		for(int r=0; r<SIFTCodeMap.rows; r++)
		{
			for(int c=0; c<SIFTCodeMap.cols; c++)
			{
				fop << SIFTCodeMap.at<float>(r,c);
				fop << " ";
			}
			fop << endl;
		}
		fop.close();
	}
	
	// extract sift features from downsampled image
			
	mvd->siftCodeMapVec.push_back(SIFTCodeMap);
}
void extractTexton(int i, struct multiViewData * mvd, string fdir)
{
	string idstr;          // string which will contain the result
	stringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'k' in the characters in the stream
	idstr = convert.str(); // set 'idstr' to the contents of the stream
	string textonFileName = dir+"textonMap_"+fdir+".txt";

	// extract texton features from downsampled image
	Mat textonMap;
	ifstream iop;
	iop.open(textonFileName);
	if(iop.is_open())
	{
		textonMap = Mat(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		for(int r=0; r<textonMap.rows; r++)
			for(int c=0; c<textonMap.cols; c++)
				iop >> textonMap.at<float>(r,c);
		iop.close();
	}
	else
	{
		iop.close();

		Mat img8;
		mvd->dsImg64Vec[i].convertTo(img8,CV_8UC3);

		CTexton textonFactory(img8, mvd->textonVocabulary[i]);
		textonFactory.extractTexton(textonMap);
		ofstream fop;
		fop.open(textonFileName);
		for(int r=0; r<textonMap.rows; r++)
		{
			for(int c=0; c<textonMap.cols; c++)
			{
				fop << textonMap.at<float>(r,c);
				fop << " ";
			}
			fop << endl;
		}
		fop.close();
	}
	
	mvd->textonMapVec.push_back(textonMap);


}

void calTextonLikelihood(vector<Mat>& textonHist, struct multiViewData * mvd, int i, 
				   vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec)
{

		Mat textonLikelihoodFG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		Mat textonLikelihoodBG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		cv::Point p;
		for(p.y=0; p.y<mvd->dsImg64Vec[i].rows; p.y++)
		{
			for(p.x=0; p.x<mvd->dsImg64Vec[i].cols; p.x++)
			{
				textonLikelihoodFG.at<float>(p) = textonHist[2*i+1].at<float>(mvd->textonMapVec[i].at<float>(p));
				textonLikelihoodBG.at<float>(p) = textonHist[2*i].at<float>(mvd->textonMapVec[i].at<float>(p));
			}
		}

		string frmid;
		stringstream sscvt;
		sscvt << i;
		frmid = sscvt.str();

		//imshow("FG texton "+frmid, textonLikelihoodFG*255.0);
		//imshow("BG texton", textonLikelihoodBG*255.0);
		//waitKey();

		textonLikelihoodFGVec.push_back(textonLikelihoodFG);
		textonLikelihoodBGVec.push_back(textonLikelihoodBG);

}

void calAllTextonLikelihood(vector<Mat>& textonHist, struct multiViewData * mvd, int i, 
				   vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec)
{

		Mat textonLikelihoodFG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		Mat textonLikelihoodBG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		cv::Point p;
		for(p.y=0; p.y<mvd->dsImg64Vec[i].rows; p.y++)
		{
			for(p.x=0; p.x<mvd->dsImg64Vec[i].cols; p.x++)
			{
				double fgP = 0, bgP = 0;
				for( int k=0; k<mvd->dsImg64Vec.size(); k++)
				{
					fgP += textonHist[2*k+1].at<float>(mvd->textonMapVec[i].at<float>(p));
					bgP += textonHist[2*k].at<float>(mvd->textonMapVec[i].at<float>(p));
				}

				textonLikelihoodFG.at<float>(p) = fgP/(double)mvd->dsImg64Vec.size();;
				textonLikelihoodBG.at<float>(p) = bgP/(double)mvd->dsImg64Vec.size();;
			}
		}

		string frmid;
		stringstream sscvt;
		sscvt << i;
		frmid = sscvt.str();

		//imshow("FG texton "+frmid, textonLikelihoodFG*255.0);
		//imshow("BG texton", textonLikelihoodBG*255.0);
		//waitKey();

		textonLikelihoodFGVec.push_back(textonLikelihoodFG);
		textonLikelihoodBGVec.push_back(textonLikelihoodBG);

}

void calTextonLikelihood(struct multiViewData * train_mvd, struct multiViewData * mvd, int i, 
				   vector<Mat>& textonLikelihoodFGVec,vector<Mat>& textonLikelihoodBGVec)
{

		Mat textonLikelihoodFG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		Mat textonLikelihoodBG(mvd->dsImg64Vec[i].rows, mvd->dsImg64Vec[i].cols, CV_32FC1);
		cv::Point p;
		for(p.y=0; p.y<mvd->dsImg64Vec[i].rows; p.y++)
		{
			for(p.x=0; p.x<mvd->dsImg64Vec[i].cols; p.x++)
			{
				textonLikelihoodFG.at<float>(p) = train_mvd->textonHist[2*i+1].at<float>(mvd->textonMapVec[i].at<float>(p));
				textonLikelihoodBG.at<float>(p) = train_mvd->textonHist[2*i].at<float>(mvd->textonMapVec[i].at<float>(p));
			}
		}

		string frmid;
		stringstream sscvt;
		sscvt << i;
		frmid = sscvt.str();

		imshow("FG texton "+frmid, textonLikelihoodFG*255.0);
		imshow("BG texton "+frmid, textonLikelihoodBG*255.0);
		//waitKey();

		textonLikelihoodFGVec.push_back(textonLikelihoodFG);
		textonLikelihoodBGVec.push_back(textonLikelihoodBG);

}

void svmTrainingData(int i, struct multiViewData * mvd, struct svm_problem& prob, struct svm_node * x_space)//, 
					 //vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec)
{
	// params of lab color histogram
	Mat img_lab;
	Mat img8;
	mvd->dsImg64Vec[i].convertTo(img8,CV_8UC3);
	cvtColor(img8, img_lab, CV_BGR2Lab);

	vector<Mat> Lab_planes;
	split( img_lab, Lab_planes );

	/// Establish the number of bins
	int histSizeLab = 23;

	/// Set the ranges ( for B,G,R) )
	float rangeL[] = { 0, 101 } ;    // 0-100
	float rangeA[] = { -127, 128 } ; // -127-127
	float rangeB[] = { -127, 128 } ; // -127-127

	const float* histRangeL = { rangeL };
	const float* histRangeA = { rangeA };
	const float* histRangeB = { rangeB };

	//bool uniformLab = true; bool accumulateLab = false;

	// Set the ranges of texton histogram
	int histSize = textonVocabSize;
	float range[] = { 0, textonVocabSize } ;
	const float* histRange = { range };

	// Set the ranges of sift codeword histogram
	int histSizeBoW = siftVocabSize;
	float rangeBoW[] = { 0, siftVocabSize } ;
	const float* histRangeBoW = { rangeBoW };

	bool uniform = true; bool accumulate = false;
	int cc = 0;

	for(int j=0; j<mvd->lCntVec[i]; j++)
	{
		/// Texton histogram

		Mat t_hist;
		/// Compute the FG histograms:
		Mat SPMask32, SPMask;
		threshold(abs(mvd->dsSpVec[i]-j),SPMask32,0,1,THRESH_BINARY_INV);
		SPMask32.convertTo(SPMask,CV_8UC1);

		calcHist( &mvd->keyFrameTextonMap[i], 1, 0, SPMask, t_hist, 1, &histSize, &histRange, uniform, accumulate );

		/// Normalize
		t_hist = t_hist/(sum(t_hist).val[0]);

		//textonHistVec.push_back(t_hist.clone());

		/// SIFT codeword histogram
		Mat s_hist;
		/// Compute the FG histograms:
		calcHist( &mvd->keyFrameSIFTCodeMap[i], 1, 0, SPMask, s_hist, 1, &histSizeBoW, &histRangeBoW, uniform, accumulate );
		/// Normalize
		s_hist = s_hist/(sum(s_hist).val[0]);

		//siftHistVec.push_back(s_hist.clone());

		/// Lab color histogram
		Mat L_hist, a_hist, b_hist;
		/// Compute the histograms:
		calcHist( &Lab_planes[0], 1, 0, SPMask, L_hist, 1, &histSizeLab, &histRangeL, uniform, accumulate );
		calcHist( &Lab_planes[1], 1, 0, SPMask, a_hist, 1, &histSizeLab, &histRangeA, uniform, accumulate );
		calcHist( &Lab_planes[2], 1, 0, SPMask, b_hist, 1, &histSizeLab, &histRangeB, uniform, accumulate );

		/// Normalize the result to [ 0, histImage.rows ]
		L_hist = L_hist/(sum(L_hist).val[0]);
		a_hist = a_hist/(sum(a_hist).val[0]);
		b_hist = b_hist/(sum(b_hist).val[0]);

		Mat labHist(3*L_hist.rows, 1, CV_32FC1);

		// concatenate lab channels
		L_hist.copyTo(labHist(cv::Rect(0, 0, L_hist.cols, L_hist.rows)));
		a_hist.copyTo(labHist(cv::Rect(0, L_hist.rows, a_hist.cols, a_hist.rows)));
		b_hist.copyTo(labHist(cv::Rect(0, L_hist.rows+a_hist.rows, b_hist.cols, b_hist.rows)));

		labHist = labHist/(sum(labHist).val[0]);
						
		//labHistVec.push_back(labHist.clone());
				
		// add label of sp center as training label
		int L = (int) 2*mvd->dsSegMask[i].at<uchar>((int)mvd->centDsSPViewVec[i][j].y, (int)mvd->centDsSPViewVec[i][j].x)-1;

		prob.y[j] = L;
		prob.x[j] = &x_space[cc];

		for (int v=0; v<textonVocabSize+3*histSizeLab+histSizeBoW; ++v)
		{
			x_space[cc].index =  v;
			if(v<textonVocabSize)
				x_space[cc].value =  t_hist.at<float>(v);
			else if(v<textonVocabSize+3*histSizeLab)
				x_space[cc].value =  labHist.at<float>(v-textonVocabSize);
			else
				x_space[cc].value =  s_hist.at<float>(v-textonVocabSize-3*histSizeLab);
			cc++;
		}
		x_space[cc++].index = -1;


	}
}


void svmTestingData(int i, int j, struct multiViewData * mvd, struct svm_node *temp_vec, 
					vector<Mat>& textonHistVec,
			vector<Mat>& labHistVec, vector<Mat>& siftHistVec)
{

		// params of lab color histogram
	Mat img_lab;
	Mat img8;
	mvd->dsImg64Vec[i].convertTo(img8,CV_8UC3);
	cvtColor(img8, img_lab, CV_BGR2Lab);

	vector<Mat> Lab_planes;
	split( img_lab, Lab_planes );

	/// Establish the number of bins
	int histSizeLab = 23;

	/// Set the ranges ( for B,G,R) )
	float rangeLab[] = { 0, 256 } ;
	const float* histRangeLab = { rangeLab };

	//bool uniformLab = true; bool accumulateLab = false;

	// Set the ranges of texton histogram
	int histSize = textonVocabSize;
	float range[] = { 0, textonVocabSize } ;
	const float* histRange = { range };

	// Set the ranges of sift codeword histogram
	int histSizeBoW = siftVocabSize;
	float rangeBoW[] = { 0, siftVocabSize } ;
	const float* histRangeBoW = { rangeBoW };

	bool uniform = true; bool accumulate = false;
	
	Mat SPMask32, SPMask;
	threshold(abs(mvd->dsSpVec[i]-j),SPMask32,0,1,THRESH_BINARY_INV);
	SPMask32.convertTo(SPMask,CV_8UC1);
	//imshow("SPMask",SPMask*255); waitKey();

	Mat t_hist;
	/// Compute the FG texton histograms:
	calcHist( &mvd->textonMapVec[i], 1, 0, SPMask, t_hist, 1, &histSize, &histRange, uniform, accumulate );

	/// Normalize
	t_hist = t_hist/(sum(t_hist).val[0]);
	//float response = SVM.predict(t_hist);

	textonHistVec.push_back(t_hist);

	/// SIFT codeword histogram
	Mat s_hist;
	/// Compute the FG histograms:
	calcHist( &mvd->siftCodeMapVec[i], 1, 0, SPMask, s_hist, 1, &histSizeBoW, &histRangeBoW, uniform, accumulate );
	/// Normalize
	s_hist = s_hist/(sum(s_hist).val[0]);

	siftHistVec.push_back(s_hist);

	/// Lab color histogram
	Mat L_hist, a_hist, b_hist;
	/// Compute the histograms:
	calcHist( &Lab_planes[0], 1, 0, SPMask, L_hist, 1, &histSizeLab, &histRangeLab, uniform, accumulate );
	calcHist( &Lab_planes[1], 1, 0, SPMask, a_hist, 1, &histSizeLab, &histRangeLab, uniform, accumulate );
	calcHist( &Lab_planes[2], 1, 0, SPMask, b_hist, 1, &histSizeLab, &histRangeLab, uniform, accumulate );


	/// Normalize the result to [ 0, histImage.rows ]
	L_hist = L_hist/(sum(L_hist).val[0]);
	a_hist = a_hist/(sum(a_hist).val[0]);
	b_hist = b_hist/(sum(b_hist).val[0]);

	Mat labHist(3*L_hist.rows, 1, CV_32FC1);
	labHistVec.push_back(labHist);

	// concatenate lab channels
	L_hist.copyTo(labHist(cv::Rect(0, 0, L_hist.cols, L_hist.rows)));
	a_hist.copyTo(labHist(cv::Rect(0, L_hist.rows, a_hist.cols, a_hist.rows)));
	b_hist.copyTo(labHist(cv::Rect(0, L_hist.rows+a_hist.rows, b_hist.cols, b_hist.rows)));
						

	// convert vector to sparse vector
	//struct svm_node *temp_vec = Malloc(struct svm_node, textonVocabSize+3*histSizeLab+histSizeBoW+1);

	for (int v=0; v<textonVocabSize+3*histSizeLab+histSizeBoW; ++v)
	{
		temp_vec[v].index =v;
		if(v<textonVocabSize)
			temp_vec[v].value =  t_hist.at<float>(v);
		else if(v<textonVocabSize+3*histSizeLab)
			temp_vec[v].value =  labHist.at<float>(v-textonVocabSize);
		else
			temp_vec[v].value =  s_hist.at<float>(v-textonVocabSize-3*histSizeLab);
	}
	temp_vec[textonVocabSize+3*histSizeLab+histSizeBoW].index=-1; // mark end of sparse vector

}

void matchingToFundamMat(struct multiViewData * mvd)
{
	//string calibfile = "calibration3.txt";
	//vector<Mat> intrinsicVec;
	//vector<Mat> distVec;
	//vector<Mat> RVec;
	//vector<Mat> TVec;
	//loadCameraParam(calibfile, intrinsicVec, distVec, RVec, TVec);

	cout << "Match feature points" << endl;
	SurfFeatureDetector detector;
	//SiftFeatureDetector detector;
	//OrbFeatureDetector detector;
	vector<vector<KeyPoint> > keypointsVec; 
	std::vector<KeyPoint> keypoints;
	vector<Mat> imgGVec;

	for( int i=0; i<mvd->img64Vec.size(); i++)
	{
		Mat imgG, img8;
		mvd->img64Vec[i].convertTo(img8, CV_8UC3);

		cvtColor(img8,imgG,CV_BGR2GRAY);
		imgGVec.push_back(imgG.clone());

		detector.detect( imgG, keypoints );
		keypointsVec.push_back(keypoints);
		keypoints.clear();
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	//SiftDescriptorExtractor extractor;
	//OrbDescriptorExtractor extractor;

	vector<Mat> descriptorsVec;

	for( int i=0; i<mvd->img64Vec.size(); i++)
	{
		Mat descriptors;
		extractor.compute( imgGVec[i], keypointsVec[i], descriptors );
		descriptorsVec.push_back(descriptors);
	}

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher< L2<float> > matcher;
	vector<vector< DMatch > > matchesVec;
	std::vector< DMatch > matches;
	//vector<Mat> fundamMatVec;
		
	// cam with larger indx located on the left hand
	for( int i=0; i<mvd->img64Vec.size()-1; i++)
	{

		matcher.match( descriptorsVec[i+1], descriptorsVec[i], matches );
		matchesVec.push_back(matches);

		// Convert 1 vector of keypoints into
		// 2 vectors of Point2f for compute F matrix
		std::vector<int> pointIndexesLeft;
		std::vector<int> pointIndexesRight;
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it) {

				// Get the indexes of the selected matched keypoints
				pointIndexesLeft.push_back(it->queryIdx);
				pointIndexesRight.push_back(it->trainIdx);
		}

		// Convert keypoints into Point2f
		std::vector<cv::Point2f> selPointsLeft, selPointsRight;
		cv::KeyPoint::convert(keypointsVec[i+1],selPointsLeft,pointIndexesLeft);
		cv::KeyPoint::convert(keypointsVec[i],selPointsRight,pointIndexesRight);

		// Compute F matrix from n>=8 matches
		cv::Mat fundemental = cv::findFundamentalMat(
			cv::Mat(selPointsLeft), // points in first image
			cv::Mat(selPointsRight), // points in second image
			CV_FM_RANSAC);       // 8-point method

		mvd->fundamMatVec.push_back(fundemental.clone());



		matches.clear();
	}

	if(mvd->img64Vec.size()>2)
	{
		int i = mvd->img64Vec.size()-1;
		matcher.match( descriptorsVec[0], descriptorsVec[i], matches );
		matchesVec.push_back(matches);
			
		// Convert 1 vector of keypoints into
		// 2 vectors of Point2f for compute F matrix
		std::vector<int> pointIndexesLeft;
		std::vector<int> pointIndexesRight;
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it) {

				// Get the indexes of the selected matched keypoints
				pointIndexesLeft.push_back(it->queryIdx);
				pointIndexesRight.push_back(it->trainIdx);
		}

		// Convert keypoints into Point2f
		std::vector<cv::Point2f> selPointsLeft, selPointsRight;
		cv::KeyPoint::convert(keypointsVec[0],selPointsLeft,pointIndexesLeft);
		cv::KeyPoint::convert(keypointsVec[i],selPointsRight,pointIndexesRight);

		// Compute F matrix from n>=8 matches
		cv::Mat fundemental = cv::findFundamentalMat(
		cv::Mat(selPointsLeft), // points in first image
		cv::Mat(selPointsRight), // points in second image
		CV_FM_RANSAC);       // 8-point method

		mvd->fundamMatVec.push_back(fundemental.clone());

		matches.clear();
	}
}


void calibToFundamMat(struct multiViewData * mvd, string& calibfile, string frmidx)
{
	//string calibfile = "calibration3.txt";

	vector<Mat> intrinsicVec;
	vector<Mat> distVec;
	vector<Mat> RVec;
	vector<Mat> TVec;
	loadCameraParam(calibfile, intrinsicVec, distVec, RVec, TVec);

	cout << "mvd->img64Vec.size() = " << mvd->img64Vec.size() << endl;

	cout << "- Perform stereo matching"	<< endl;
	// cam with larger indx located on the left hand
	for( int i=0; i<mvd->img64Vec.size()-1; i++)
	{
		cout << "  - View"	<< i << endl;
		double t = (double)cv::getTickCount();

		Mat imgL64 = mvd->img64Vec[i+1];
		Mat imgR64 = mvd->img64Vec[i];

		Mat imgL, imgR;
		imgL64.convertTo(imgL,CV_8UC3);
		imgR64.convertTo(imgR,CV_8UC3);
		
		Mat R1T;
		transpose(RVec[i+1], R1T);
		Mat R = RVec[i]*R1T;
		Mat T = (TVec[i]-R*TVec[i+1])*1.0;
		Mat M1 = intrinsicVec[i+1];
		Mat M2 = intrinsicVec[i];
		Mat D1 = distVec[i+1];
		Mat D2 = distVec[i];

		Mat F;
		calFundamMat(M1, M2, R, T, F);
		mvd->fundamMatVec.push_back(F.clone());

		// rectification of image pairs
		Mat l_rect, r_rect, H1, H2, tML, tMR, Q;
		rectifyStereoImage(imgL, imgR, M1, M2, R, T, D1, 
						D2, l_rect, r_rect, H1, H2, tML, tMR, Q);


		Mat l_HWarp = tML*H1;

		// downsampling to speed up stereo matching
		Mat r_rect_ds, l_rect_ds;
		cv::resize(l_rect, l_rect_ds, cv::Size(imgR.cols/6.0,imgR.rows/6.0), CV_INTER_CUBIC);
		cv::resize(r_rect, r_rect_ds, cv::Size(imgR.cols/6.0,imgR.rows/6.0), CV_INTER_CUBIC);

		string idstr;          // string which will contain the result
		stringstream convert;   // stream used for the conversion
		convert << i+1;      // insert the textual representation of 'k' in the characters in the stream
		idstr = convert.str(); // set 'idstr' to the contents of the stream

		string dispFileName = dir+"dispMap_f"+frmidx+"c"+idstr+".png";
		Mat dispMat8L = imread(dispFileName, 0);

		string rectDispFileName = dir+"rectDispMap_f"+frmidx+"c"+idstr+".png";
		Mat dispUS8 = imread(rectDispFileName, 0);

		if(!dispMat8L.data || !dispUS8.data)
		{
			//cout << "l_rect_ds.cols = " << l_rect_ds.cols << endl;
			//imshow("l_rect_ds",l_rect_ds);
			//imshow("r_rect_ds",r_rect_ds);
			//waitKey();

			Mat dispMat8;
			stereoMatchingMRF(l_rect_ds, r_rect_ds, dispMat8);

			//Mat dispUS8;
			cv::resize(dispMat8,dispUS8,cv::Size(l_rect.cols, l_rect.rows), CV_INTER_NN);
			imwrite(rectDispFileName,dispUS8);

			///*
			
			cv::warpPerspective(dispUS8, dispMat8L, l_HWarp.inv(), cv::Size(imgL.cols, imgL.rows), 
				INTER_NEAREST+CV_WARP_FILL_OUTLIERS);
			//*/

			imwrite(dispFileName,dispMat8L);

			//Mat dsDispMat8;

		}
		
		Mat dsDispMat8;
		cv::resize(dispMat8L, dsDispMat8, cv::Size(mvd->dsImg64Vec[0].cols,mvd->dsImg64Vec[0].rows), CV_INTER_NN);
		//cv::resize(dispMat8L, dsDispMat8, cv::Size(mvd->dsImgVec[0].cols,mvd->dsImgVec[0].rows), CV_INTER_LINEAR);
		mvd->disparityMapVec.push_back(dsDispMat8);

		//cv::namedWindow("disp"+idstr, CV_WINDOW_NORMAL);
		//cv::imshow("disp"+idstr, dsDispMat8);

		t = (double)cv::getTickCount() - t;
		cout << "     " << t/cv::getTickFrequency() << " seconds on stereo matching" << endl;

		//waitKey();

	}

	if(mvd->img64Vec.size()>2)
	{
		int i = mvd->img64Vec.size()-1;
		cout << "  - View"	<< i << endl;
		double t = (double)cv::getTickCount();

		Mat imgL64 = mvd->img64Vec[0];
		Mat imgR64 = mvd->img64Vec[i];

		Mat imgL, imgR;
		imgL64.convertTo(imgL,CV_8UC3);
		imgR64.convertTo(imgR,CV_8UC3);
		
		Mat R1T;
		transpose(RVec[0], R1T);
		Mat R = RVec[i]*R1T;
		Mat T = (TVec[i]-R*TVec[0])*1.0;
		Mat M1 = intrinsicVec[0];
		Mat M2 = intrinsicVec[i];
		Mat D1 = distVec[0];
		Mat D2 = distVec[i];

		
		Mat F;
		calFundamMat(M1, M2, R, T, F);
		mvd->fundamMatVec.push_back(F.clone());
		

		// rectification of image pairs
		Mat l_rect, r_rect, H1, H2, tML, tMR, Q;
		rectifyStereoImage(imgL, imgR, M1, M2, R, T, D1, 
						D2, l_rect, r_rect, H1, H2, tML, tMR, Q);

		Mat l_HWarp = tML*H1;

		// downsampling to speed up stereo matching
		Mat r_rect_ds, l_rect_ds;
		cv::resize(l_rect, l_rect_ds, cv::Size(imgR.cols/6.0,imgR.rows/6.0), CV_INTER_CUBIC);
		cv::resize(r_rect, r_rect_ds, cv::Size(imgR.cols/6.0,imgR.rows/6.0), CV_INTER_CUBIC);

		string idstr;          // string which will contain the result
		stringstream convert;   // stream used for the conversion
		convert << 0;      // insert the textual representation of 'k' in the characters in the stream
		idstr = convert.str(); // set 'idstr' to the contents of the stream

		string dispFileName = dir+"dispMap_f"+frmidx+"c"+idstr+".png";
		Mat dispMat8L = imread(dispFileName, 0);

		string rectDispFileName = dir+"rectDispMap_f"+frmidx+"c"+idstr+".png";
		Mat dispUS8 = imread(rectDispFileName, 0);

		if(!dispMat8L.data || !dispUS8.data)
		{
			//cout << "l_rect_ds.cols = " << l_rect_ds.cols << endl;
			//imshow("l_rect_ds",l_rect_ds);
			//imshow("r_rect_ds",r_rect_ds);
			//waitKey();

			Mat dispMat8;
			stereoMatchingMRF(l_rect_ds, r_rect_ds, dispMat8);

			//Mat dispUS8;
			cv::resize(dispMat8,dispUS8,cv::Size(l_rect.cols, l_rect.rows), CV_INTER_NN);
			imwrite(rectDispFileName,dispUS8);

			///*
			//Mat dispMat8L;
			cv::warpPerspective(dispUS8, dispMat8L, l_HWarp.inv(), cv::Size(imgL.cols, imgL.rows), 
				INTER_NEAREST+CV_WARP_FILL_OUTLIERS);
			//*/

			imwrite(dispFileName,dispMat8L);

			//Mat dsDispMat8;

		}

		Mat dsDispMat8;
		cv::resize(dispMat8L, dsDispMat8, cv::Size(mvd->dsImg64Vec[0].cols,mvd->dsImg64Vec[0].rows), CV_INTER_NN);
		mvd->disparityMapVec.push_back(dsDispMat8);

		//cv::namedWindow("disp"+idstr, CV_WINDOW_NORMAL);
		//cv::imshow("disp"+idstr, dsDispMat8);
		//Mat xyz;
        //reprojectImageTo3D(dispMat, xyz, Q, true);

		//imshow("point cloud", xyz);
		//waitKey();

		t = (double)cv::getTickCount() - t;
		cout << "     " << t/cv::getTickFrequency() << " seconds on stereo matching" << endl;
	}
	
}


void stereoMatchingMRF(Mat& imgL, Mat& imgR, Mat& dispMat8)
{

	// MRF based stereo matching framework
	enum algtypes {aICM, aExpansion, aSwap, aTRWS, aBPS, aBPM};

	int MAXITER = 5;
	int outerIter, innerIter;
	int noChange = 0;
	MRF *mrf = NULL;
	int numAlg = aExpansion;

	int nD = 256;           // disparity levels (d = 0 .. nD-1)
	int birchfield = 0;    // use Birchfield/Tomasi costs
	int squaredDiffs = 0;  // use squared differences (absolute differences by default)
	int truncDiffs = 255;  // truncated differences (before squaring), by default not
	int MRFalg = 1;        // 0-ICM, 1-GC/expansion (default), 2-GC/swap, 3-TRWS, 4-BPS, 5-BPM, 9-all
	int smoothexp = 1;     // exponent of smoothness term: 1 (default) or 2, i.e. L1 or L2 norm
	int smoothmax = 100;     // maximum value of smoothness term (2 by default)
	int lambda = 1;       // weight of smoothness term (20 by default)
	int gradThresh = 2;   // intensity gradient cue threshold, by default none (-1)
	int gradPenalty = 2;   // if grad < gradThresh, multiply smoothness cost by this
	int outscale = -1;     // scale factor for disparities; -1 means full range 255.0/(nD-1)
	int writeTimings = 0;  // write timings to dispL.csv

	//cout << "imgL.step1() = " << imgL.elemSize() << endl;

	uchar* bufferL = new uchar [imgL.rows * imgL.cols * imgL.elemSize()];
	memcpy(bufferL, imgL.data, imgL.rows * imgL.cols * imgL.elemSize());
	uchar* bufferR = new uchar [imgL.rows * imgL.cols * imgL.elemSize()];
	memcpy(bufferR, imgR.data, imgL.rows * imgL.cols * imgL.elemSize());

	CByteImage im1, im2;      // input images (gray or color)
	CByteImage disp;
	CShape s(imgL.cols, imgL.rows, 3);
	int width = imgL.cols;
	int height = imgL.rows;

	im1.ReAllocate(s, bufferL, true, 3*imgL.cols);
	im2.ReAllocate(s, bufferR, true, 3*imgL.cols);

	MRF::CostVal *dsi = NULL;
	computeDSI(im1, im2, dsi, nD, birchfield, squaredDiffs, truncDiffs);
	DataCost *dcost = new DataCost(dsi);

	SmoothnessCost *scost;
	MRF::CostVal *hCue = NULL, *vCue = NULL;
	if (gradThresh > 0) {
		computeCues(im1, hCue, vCue, gradThresh, gradPenalty);
		scost = new SmoothnessCost(smoothexp, smoothmax, lambda, hCue, vCue);
	} else {
		scost = new SmoothnessCost(smoothexp, smoothmax, lambda);
	}

	EnergyFunction *energy = new EnergyFunction(dcost, scost);

	outerIter = MAXITER;
	innerIter = 1;
	//if (MRFalg < 9 && numAlg != MRFalg) continue;
	switch (numAlg) 
	{
		case aICM:       mrf = new ICM(width, height, nD, energy); innerIter = 5; break;
		case aExpansion: mrf = new Expansion(width, height, nD, energy); break;
		case aSwap:      mrf = new Swap(width, height, nD, energy); break;
		case aTRWS:      mrf = new TRWS(width, height, nD, energy); break;
		case aBPS:       mrf = new BPS(width, height, nD, energy); break;
		case aBPM:       mrf = new MaxProdBP(width, height, nD, energy); outerIter = MAXITER/2; break;
		default: throw new CError("unknown algorithm number");
	}

	mrf->initialize();

	bool initializeToWTA = (numAlg == aICM); 
	if (initializeToWTA) {
		WTA(dsi, width, height, nD, disp);
		setDisparities(disp, mrf);
	} else {
		mrf->clearAnswer();
	}

	MRF::EnergyVal E, Ed, Es, Eold;
	Ed = mrf->dataEnergy();
	Es = mrf->smoothnessEnergy();
	E = Ed + Es; // mrf->totalEnergy();
	Eold = E;


	float t, tot_t = 0;
	double lowerBound = 0;
	int iter;
	for (iter = 0; iter < outerIter; iter++) 
	{
		cout << "   - iter = " << iter << endl;
		mrf->optimize(innerIter, t);
		tot_t += t ;
    
		Ed = mrf->dataEnergy();
		Es = mrf->smoothnessEnergy();
		E = Ed + Es; // mrf->totalEnergy();
		if (numAlg == aTRWS)
			lowerBound = mrf->lowerBound();

		if (E == Eold) 
		{
			if (numAlg <= aSwap) // ICM, Expansion and Swap converge
				break;
			noChange++;
			if (noChange >= 10) // if energy hasn't changed for 10 iterations, it's save to quit
				break;
		} 
		else
			noChange = 0;

		Eold = E;
	}

	getDisparities(mrf, width, height, disp);

	Mat dispMat(height, width, CV_8UC1, (uchar*)&disp.Pixel(0, 0, 0));

	dispMat8 = dispMat.clone();

	delete [] dsi;
	delete [] hCue;
	delete [] vCue;

	//delete [] bufferL;
	//delete [] bufferR;

	delete mrf;
	delete energy;
	delete scost;
	delete dcost;
	
	/*
	double minVal, maxVal;
	minMaxLoc(dispMat, &minVal, &maxVal);
	Mat visMap = (dispMat-minVal)/(double)(maxVal-minVal)*255.0;
	imshow("original disp", visMap);//*/

	//imshow("original disp", dispMat8);
	//waitKey();
}

void calFundamMat(Mat& M1, Mat& M2, Mat& R, Mat& T, Mat& F)
{
	// Compute F matrix from calibration
	Mat term0 = M2.inv();
	Mat term1;
	transpose(term0, term1);

	Mat term2 = Mat::zeros(3,3,CV_64FC1);

	term2.at<double>(0,1) = -T.at<double>(2,0);
	term2.at<double>(1,0) = T.at<double>(2,0);

	term2.at<double>(0,2) = T.at<double>(1,0);
	term2.at<double>(2,0) = -T.at<double>(1,0);

	term2.at<double>(1,2) = -T.at<double>(0,0);
	term2.at<double>(2,1) = T.at<double>(0,0);

	Mat term3 = M1.inv();

	F = term1*term2*R*term3;
	
}
void loadCameraParam(string& calibfile, vector<Mat>& intrinsicVec, vector<Mat>& distVec,
					 vector<Mat>& RVec, vector<Mat>& TVec)
{
	//string calibfile = "calibration3.txt";
	ifstream fid;
	fid.open(calibfile);

	if (fid.is_open()) {
		//vector<Mat> intrinsicVec;
		//vector<double> distVec; //distortion param
		//vector<Mat> extrinsicVec;
		//vector<Mat> RVec;
		//vector<Mat> TVec;

		int nCams, distM, minRow, maxRow, minCol, maxCol;
		//double distortion;
		fid >> nCams;
		fid >> distM;

		
		while (!fid.eof()) {

			Mat intrinsicMat = Mat::zeros(3,3,CV_64FC1);
			Mat RMat(3,3,CV_64FC1);
			Mat TMat(3,1,CV_64FC1);
			Mat DMat = Mat::zeros(8,1,CV_64FC1);

			fid >> minRow;
			fid >> maxRow;
			fid >> minCol;
			fid >> maxCol;

			fid >> intrinsicMat.at<double>(0,0);//fx
			fid >> intrinsicMat.at<double>(1,1);//fy
			fid >> intrinsicMat.at<double>(0,2);//cx
			fid >> intrinsicMat.at<double>(1,2);//cy
			intrinsicMat.at<double>(2,2) = 1;

			fid >> DMat.at<double>(0,0);
			distVec.push_back(DMat);

			//cout << "Rotation" << endl;
			for(int i=0;i<3;i++)
			{
				for(int j=0;j<3;j++)
				{
					fid >> RMat.at<double>(i,j);
					//cout << RMat.at<double>(i,j) << " ";
				}
				//cout << endl;
			}

			//cout << "Translation" << endl;
			for(int i=0;i<3;i++)
			{
				fid >> TMat.at<double>(i,0);
				//cout << TMat.at<double>(i,0) << " ";
			}
			//cout << endl;

			intrinsicVec.push_back(intrinsicMat.clone());
			RVec.push_back(RMat.clone());
			TVec.push_back(TMat.clone());

		}
	}
	fid.close();
		//Mat R = RVec[0]*(RVec[1].inv());
		//Mat T = TVec[0]-TVec[1];

}

void rectifyStereoImage(Mat& imgL, Mat& imgR, Mat& M1, Mat& M2, Mat& R, Mat& T, Mat& D1, 
						Mat& D2, Mat& l_rect, Mat& r_rect, Mat& H1, Mat& H2, Mat& tML, Mat& tMR, Mat& Q)
{
	Mat R1, P1, R2, P2;

	cv::Size img_size = imgL.size();

	stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, 
		CALIB_ZERO_DISPARITY, -1, img_size, 0, 0 );

	// align the rectified image so that they are all in the image plane
	double tlptArr[] = {0,0,1};
	Mat tlpt(3,1,CV_64FC1,tlptArr);
	double trptArr[] = {imgL.cols,0,1};
	Mat trpt(3,1,CV_64FC1,trptArr);
	double blptArr[] = {0,imgL.rows,1};
	Mat blpt(3,1,CV_64FC1,blptArr);
	double brptArr[] = {imgL.cols,imgL.rows,1};
	Mat brpt(3,1,CV_64FC1,brptArr);

	// P1 represents the new cam mat 
	Mat P1r = P1.colRange(0,3);
	// 1. project pixel back to 3D space; 2. rotate to new cam; 3. project on rectified image plane
	H1 = P1r*R1*M1.inv();

	// new projections of four vertices
	Mat LtlptWarp = H1*tlpt;
	Mat LtrptWarp = H1*trpt;
	Mat LblptWarp = H1*blpt;
	Mat LbrptWarp = H1*brpt;

	LtlptWarp = LtlptWarp/LtlptWarp.at<double>(2);
	LtrptWarp = LtrptWarp/LtrptWarp.at<double>(2);
	LblptWarp = LblptWarp/LblptWarp.at<double>(2);
	LbrptWarp = LbrptWarp/LbrptWarp.at<double>(2);

	// TL and BR vertices of the bounding box of warped image
	double LtlOrig[] = {min(LtlptWarp.at<double>(0), LblptWarp.at<double>(0)), 
						min(LtlptWarp.at<double>(1), LtrptWarp.at<double>(1))};

	double LbrOrig[] = {max(LtrptWarp.at<double>(0), LbrptWarp.at<double>(0)), 
						max(LblptWarp.at<double>(1), LbrptWarp.at<double>(1))}; 

	// similar on the right image
	Mat P2r = P2.colRange(0,3);
	H2 = P2r*R2*M2.inv();

	Mat RtlptWarp = H2*tlpt;
	Mat RtrptWarp = H2*trpt;
	Mat RblptWarp = H2*blpt;
	Mat RbrptWarp = H2*brpt;
	RtlptWarp = RtlptWarp/RtlptWarp.at<double>(2);
	RtrptWarp = RtrptWarp/RtrptWarp.at<double>(2);
	RblptWarp = RblptWarp/RblptWarp.at<double>(2);
	RbrptWarp = RbrptWarp/RbrptWarp.at<double>(2);

	double RtlOrig[] = {min(RtlptWarp.at<double>(0), RblptWarp.at<double>(0)), 
						min(RtlptWarp.at<double>(1), RtrptWarp.at<double>(1))};

	double RbrOrig[] = {max(RtrptWarp.at<double>(0), RbrptWarp.at<double>(0)), 
						max(RblptWarp.at<double>(1), RbrptWarp.at<double>(1))}; 

	/*
	double deltaR, deltaL;
	if(LtlOrig[0]<RtlOrig[0])
	{
		deltaR = RtlOrig[0]-LtlOrig[0];
		deltaL = 0;
	}
	else if(LtlOrig[0]>RtlOrig[0])
	{
		deltaL = LtlOrig[0]-RtlOrig[0];
		deltaR = 0;
	}

	// new TL and BR vertices after translation/sliding
	double tlptWarp[] = {min(LtlOrig[0]-deltaL, RtlOrig[0]-deltaR), min(LtlOrig[1], RtlOrig[1])};
	double brptWarp[] = {max(LbrOrig[0]-deltaL, RbrOrig[0]-deltaR), max(LbrOrig[1], RbrOrig[1])};*/

	double tlptWarp[] = {min(LtlOrig[0], RtlOrig[0]), min(LtlOrig[1], RtlOrig[1])};
	double brptWarp[] = {max(LbrOrig[0], RbrOrig[0]), max(LbrOrig[1], RbrOrig[1])};

	int widthWarp = brptWarp[0]-tlptWarp[0];
	int heightWarp = brptWarp[1]-tlptWarp[1];

	// translation after warp
	tML = (Mat_<double>(3,3) << 1, 0, -tlptWarp[0],
									0, 1, -tlptWarp[1],
									0, 0, 1);
	/*
	Mat tML = (Mat_<double>(3,3) << 1, 0, -tlptWarp[0]-deltaL,
									0, 1, -tlptWarp[1],
									0, 0, 1);*/
		
	// transmation matrix consisting of 1. warpping 2. translation
	Mat l_HWarp = tML*H1;
	//Mat l_rect;
	warpPerspective(imgL, l_rect, l_HWarp, cv::Size(widthWarp,heightWarp), 
		INTER_LINEAR+CV_WARP_FILL_OUTLIERS);

	//namedWindow("l_rect", CV_WINDOW_NORMAL);
	//imshow("l_rect",l_rect);

	/*
	Mat tMR = (Mat_<double>(3,3) << 1, 0, -tlptWarp[0]-deltaR,
							0, 1, -tlptWarp[1],
							0, 0, 1);*/
	tMR = (Mat_<double>(3,3) << 1, 0, -tlptWarp[0],
							0, 1, -tlptWarp[1],
							0, 0, 1);

	Mat r_HWarp = tMR*H2;
	//Mat r_rect;
	warpPerspective(imgR, r_rect, r_HWarp, cv::Size(widthWarp,heightWarp), 
		INTER_LINEAR+CV_WARP_FILL_OUTLIERS);

	//namedWindow("r_rect", CV_WINDOW_NORMAL);
	//imshow("r_rect",r_rect);
}

void svmClassification(struct persist_model *rv, int i, Mat& fgProbMap, Mat& bgProbMap, struct multiViewData * mvd,
					   vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec, string fdir)
{
	string idstr;          // string which will contain the result
	stringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'k' in the characters in the stream
	idstr = convert.str(); // set 'idstr' to the contents of the stream
	string fgProbMapFileName = fdir+"_fgProbMap.txt";
	string bgProbMapFileName = fdir+"_bgProbMap.txt";

	//vector<float> SPFGProbVec(mvd->lCntVec[i],0);
	//vector<float> SPBGProbVec(mvd->lCntVec[i],0);

	ifstream iopFG, iopBG;
	iopFG.open(fgProbMapFileName);
	iopBG.open(bgProbMapFileName);
	if(iopFG.is_open() && iopBG.is_open())
	{
		for(int j=0; j<mvd->lCntVec[i]; j++)
		{
			// to load histogram vectors
			struct svm_node *temp_vec = Malloc(struct svm_node, textonVocabSize+3*histSizeLab+histSizeBoW+1);	
			svmTestingData(i, j, mvd, temp_vec, textonHistVec, labHistVec, siftHistVec);
			free(temp_vec);

			iopFG >> SPFGProbVec[j];
			iopBG >> SPBGProbVec[j];
		}
		for(int r=0; r<fgProbMap.rows; r++)
		{
			for(int c=0; c<fgProbMap.cols; c++)
			{
				iopFG >> fgProbMap.at<float>(r,c);
				iopBG >> bgProbMap.at<float>(r,c);
			}
		}

		iopFG.close();
		iopBG.close();
	}
	else
	{
		iopFG.close();
		iopBG.close();

		ofstream fopFG, fopBG;
		fopFG.open(fgProbMapFileName);
		fopBG.open(bgProbMapFileName);

		for(int j=0; j<mvd->lCntVec[i]; j++)
		{
			// convert vector to sparse vector
			struct svm_node *temp_vec = Malloc(struct svm_node, textonVocabSize+3*histSizeLab+histSizeBoW+1);
					
			svmTestingData(i, j, mvd, temp_vec, textonHistVec, labHistVec, siftHistVec);

			// apply model to vector
			double * prob_estimates = new double[2];
			double predict_label = svm_predict_probability(rv->mod,temp_vec,prob_estimates);

			free(temp_vec);

			SPFGProbVec[j] = prob_estimates[1];
			SPBGProbVec[j] = prob_estimates[0];

			fopFG << SPFGProbVec[j];
			fopFG << " ";
			fopBG << SPBGProbVec[j];
			fopBG << " ";

			delete prob_estimates;
		}


		cv::Point p;
		for(p.y=0; p.y<fgProbMap.rows; p.y++)
		{
			for(p.x=0; p.x<fgProbMap.cols; p.x++)
			{
				fgProbMap.at<float>(p) = SPFGProbVec[mvd->dsSpVec[i].at<float>(p)];
				bgProbMap.at<float>(p) = SPBGProbVec[mvd->dsSpVec[i].at<float>(p)];

				fopFG << fgProbMap.at<float>(p);
				fopFG << " ";
				fopBG << bgProbMap.at<float>(p);
				fopBG << " ";
			}
		}

		fopFG.close();
		fopBG.close();
	}



}


void svmAllClassification(vector<struct persist_model *> rvec, int i, Mat& fgProbMap, Mat& bgProbMap, struct multiViewData * mvd,
					   vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec, string fdir)
{
	string idstr;          // string which will contain the result
	stringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'k' in the characters in the stream
	idstr = convert.str(); // set 'idstr' to the contents of the stream
	string fgProbMapFileName = fdir+"_fgProbMap.txt";
	string bgProbMapFileName = fdir+"_bgProbMap.txt";

	//vector<float> SPFGProbVec(mvd->lCntVec[i],0);
	//vector<float> SPBGProbVec(mvd->lCntVec[i],0);

	ifstream iopFG, iopBG;
	iopFG.open(fgProbMapFileName);
	iopBG.open(bgProbMapFileName);
	if(iopFG.is_open() && iopBG.is_open())
	{
		for(int j=0; j<mvd->lCntVec[i]; j++)
		{
			// to load histogram vectors
			struct svm_node *temp_vec = Malloc(struct svm_node, textonVocabSize+3*histSizeLab+histSizeBoW+1);	
			svmTestingData(i, j, mvd, temp_vec, textonHistVec, labHistVec, siftHistVec);
			free(temp_vec);

			iopFG >> SPFGProbVec[j];
			iopBG >> SPBGProbVec[j];
		}
		for(int r=0; r<fgProbMap.rows; r++)
		{
			for(int c=0; c<fgProbMap.cols; c++)
			{
				iopFG >> fgProbMap.at<float>(r,c);
				iopBG >> bgProbMap.at<float>(r,c);
			}
		}

		iopFG.close();
		iopBG.close();
	}
	else
	{
		iopFG.close();
		iopBG.close();

		ofstream fopFG, fopBG;
		fopFG.open(fgProbMapFileName);
		fopBG.open(bgProbMapFileName);

		for(int j=0; j<mvd->lCntVec[i]; j++)
		{
			// convert vector to sparse vector
			struct svm_node *temp_vec = Malloc(struct svm_node, textonVocabSize+3*histSizeLab+histSizeBoW+1);
					
			svmTestingData(i, j, mvd, temp_vec, textonHistVec, labHistVec, siftHistVec);

			// apply model to vector
			double fgP = 0, bgP = 0;
			for(int k=0; k<mvd->dsImg64Vec.size(); k++)
			{
				//double * prob_estimates = new double[2];
				double prob_estimates[2];
				double predict_label = svm_predict_probability(rvec[k]->mod,temp_vec,prob_estimates);
				fgP += prob_estimates[1];
				bgP += prob_estimates[0];
			}

			free(temp_vec);

			SPFGProbVec[j] = fgP/(double)mvd->dsImg64Vec.size();
			SPBGProbVec[j] = bgP/(double)mvd->dsImg64Vec.size();

			fopFG << SPFGProbVec[j];
			fopFG << " ";
			fopBG << SPBGProbVec[j];
			fopBG << " ";

			//delete prob_estimates;
		}


		cv::Point p;
		for(p.y=0; p.y<fgProbMap.rows; p.y++)
		{
			for(p.x=0; p.x<fgProbMap.cols; p.x++)
			{
				fgProbMap.at<float>(p) = SPFGProbVec[mvd->dsSpVec[i].at<float>(p)];
				bgProbMap.at<float>(p) = SPBGProbVec[mvd->dsSpVec[i].at<float>(p)];

				fopFG << fgProbMap.at<float>(p);
				fopFG << " ";
				fopBG << bgProbMap.at<float>(p);
				fopBG << " ";
			}
		}

		fopFG.close();
		fopBG.close();
	}



}

void svmTraining(struct persist_model *rv, struct svm_parameter & param, int i, struct multiViewData * mvd,
				 //vector<Mat>& textonHistVec, vector<Mat>& labHistVec, vector<Mat>& siftHistVec,
					   int histSizeLab, int histSizeBoW)//, vector<float>& SPFGProbVec, vector<float>& SPBGProbVec)
{
	//vector<float> SPFGProbVec(mvd->lCntVec[i],0);
	//vector<float> SPBGProbVec(mvd->lCntVec[i],0);
	string idstr;          // string which will contain the result
	stringstream convert;   // stream used for the conversion
	convert << i;      // insert the textual representation of 'k' in the characters in the stream
	idstr = convert.str(); // set 'idstr' to the contents of the stream

	string model_file_name = dir+"svm_model"+idstr+".txt";
	
	std::ifstream ifile;
	ifile.open(model_file_name);
	//struct svm_model * model;

	struct svm_problem prob;        // set by read_problem
	prob.l = mvd->lCntVec[i];
	int elements = (textonVocabSize+3*histSizeLab+histSizeBoW+1)*prob.l;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	struct svm_node * x_space = Malloc(struct svm_node,elements);

	// convert input matrices to array of sparse vectors
	svmTrainingData(i, mvd, prob, x_space);//, textonHistVec, labHistVec, siftHistVec);

	if (ifile) {

		// load existing model
		rv->mod = svm_load_model(model_file_name.c_str());
		//memcpy(rv->mod,model,sizeof(struct svm_model));			
		ifile.close();
	}
	else
	{
		ifile.close();
		// setup svm_problem 
		// learn model 
		rv->mod = svm_train(&prob, &param);
		svm_save_model(model_file_name.c_str(), rv->mod);
		//memcpy(rv->mod,model,sizeof(struct svm_model));
		free(prob.x);
		free(prob.y);
	}

	/*
	for(int j=0; j<mvd->lCntVec[i]; j++)
	{
		// definite FG or BG from keyframe used for segmentation
		SPFGProbVec[j] = mvd->dsSegMask[i].at<uchar>((int)mvd->centDsSPViewVec[i][j].y, (int)mvd->centDsSPViewVec[i][j].x);
		SPBGProbVec[j] = 1-mvd->dsSegMask[i].at<uchar>((int)mvd->centDsSPViewVec[i][j].y, (int)mvd->centDsSPViewVec[i][j].x);
	}*/

	//rv->mod = model;
	
	
}


void spDisparityMap(struct multiViewData * mvd)
{
	for(int i=0; i<mvd->dsSpVec.size(); i++)
	{

		int dispIndx = i>0 ? i-1 : mvd->dsSpVec.size()-1;

		if(mvd->dsSpVec.size()>2 || i>0)
		{
			vector<int> spDisp;
			
			for(int j=0; j<mvd->lCntVec[i]; j++)
			{
				Mat SPMask32, SPMask;
				threshold(abs(mvd->dsSpVec[i]-j),SPMask32,0,1,THRESH_BINARY_INV);
				SPMask32.convertTo(SPMask,CV_8UC1);
				//cout << "dispIndx = " << dispIndx << endl;
				//cout << "mvd->disparityMapVec.size() = " << mvd->disparityMapVec.size() << endl;
				//imshow("SPMask",SPMask*255); waitKey();
				cv::Scalar avgDisp = mean(mvd->disparityMapVec[dispIndx],SPMask);
				spDisp.push_back(avgDisp.val[0]);
			}
			mvd->spDispVec.push_back(spDisp);
		}
		
	}
}

void projectDispTo3D(Mat& dispUS8, Mat& imgL, Mat& Q, Mat& l_rect, Mat& tML, Mat& H1)
{
	// translate it back to be same as from the rectification function in OpenCV
	Mat ptCloudMap(l_rect.rows, l_rect.cols, CV_32FC3);
	cv::Point pt;
	for(pt.y=0; pt.y<l_rect.rows; pt.y++)
	{
		for(pt.x=0; pt.x<l_rect.cols; pt.x++)
		{
			// x in 3d
			ptCloudMap.at<Vec3f>(pt)[0] = Q.at<float>(0,0)*(pt.x-tML.at<float>(0,2))+Q.at<float>(0,1)*(pt.y-tML.at<float>(1,2))
										+Q.at<float>(0,2)*dispUS8.at<uchar>(pt)+Q.at<float>(0,3);
			ptCloudMap.at<Vec3f>(pt)[1] = Q.at<float>(1,0)*(pt.x-tML.at<float>(0,2))+Q.at<float>(1,1)*(pt.y-tML.at<float>(1,2))
										+Q.at<float>(1,2)*dispUS8.at<uchar>(pt)+Q.at<float>(1,3);
			ptCloudMap.at<Vec3f>(pt)[2] = Q.at<float>(2,0)*(pt.x-tML.at<float>(0,2))+Q.at<float>(2,1)*(pt.y-tML.at<float>(1,2))
										+Q.at<float>(2,2)*dispUS8.at<uchar>(pt)+Q.at<float>(2,3);
			float W = Q.at<float>(3,0)*(pt.x-tML.at<float>(0,2))+Q.at<float>(3,1)*(pt.y-tML.at<float>(1,2))
										+Q.at<float>(3,2)*dispUS8.at<uchar>(pt)+Q.at<float>(3,3);
			///TODO: check infinit values here
			ptCloudMap.at<Vec3f>(pt)[0] /= W;
			ptCloudMap.at<Vec3f>(pt)[1] /= W;
			ptCloudMap.at<Vec3f>(pt)[2] /= W;

			cout << "Q.at<float>(0,0)" << Q.at<float>(0,0) << endl;
			cout << "Q.at<float>(0,1)" << Q.at<float>(0,1) << endl;
			cout << "Q.at<float>(0,2)" << Q.at<float>(0,2) << endl;

			cout << "ptCloudMap.at<Vec3f>(pt)[0]" << ptCloudMap.at<Vec3f>(pt)[0] << endl;
			cout << "ptCloudMap.at<Vec3f>(pt)[1]" << ptCloudMap.at<Vec3f>(pt)[1] << endl;
			cout << "ptCloudMap.at<Vec3f>(pt)[2]" << ptCloudMap.at<Vec3f>(pt)[2] << endl;

		}
	}

	Mat ptCloudMapUnwarped;
	cv::warpPerspective(ptCloudMap, ptCloudMapUnwarped, H1.inv(), cv::Size(imgL.cols, imgL.rows), 
			INTER_NEAREST+CV_WARP_FILL_OUTLIERS);



	//Mat xyz, xyzOrig;
    //reprojectImageTo3D(dispUS8, xyz, Q, true);

	namedWindow("point cloud", CV_WINDOW_NORMAL);
	imshow("point cloud", ptCloudMapUnwarped*255.0);
	waitKey();
}

void genSP(struct multiViewData * mvd, int i, int frameID)
{

	int width = mvd->img64Vec[i].cols; 
	int height = mvd->img64Vec[i].rows;

	string camidx;
	stringstream strconvt;
	strconvt << i;
	camidx = strconvt.str();

	string frmidx;
	stringstream strconvt2;
	strconvt2 << frameID;
	frmidx = strconvt2.str();

	Mat img8;
	mvd->dsImg64Vec[i].convertTo(img8,CV_8UC3);


	double t = (double)cv::getTickCount();

	string spFileName = dir+"sp_f"+frmidx+"c"+camidx+".txt";

	ifstream spIFile;
	spIFile.open(spFileName);

	if(!spIFile.is_open())
	{
		spIFile.close();
		// extract superpixels		
		CSuperPixel* pSP = new CSuperPixel( img8, m_spcount, m_compactness );
		Mat spMap = pSP->getSPMap();
		ofstream spOFile;
		spOFile.open(spFileName);
		cv::Point p;
		spOFile << pSP->mLabelCnt << endl;
		for(p.y=0; p.y<img8.rows; p.y++)
			for(p.x=0; p.x<img8.cols; p.x++)
				spOFile << spMap.at<float>(p) << " ";

		spOFile.close();

		Mat spClMat8;
		pSP->spClMat.convertTo(spClMat8,CV_8UC3);
		imwrite(dir+"sp_f"+frmidx+"c"+camidx+".png", spClMat8);

		//t = (double)cv::getTickCount() - t;
		//cout << "    " << t/cv::getTickFrequency() << " seconds on SP extraction" << endl;

		mvd->dsSpVec.push_back(pSP->getSPMap());
		mvd->lCntVec.push_back(pSP->mLabelCnt);
		mvd->spClVec.push_back(pSP->spClMat);
		mvd->tolLabel += pSP->mLabelCnt;

		Mat SPContour = pSP->getSPContour();
		//SPContourVec.push_back(SPContour);

		imwrite(dir+"Superpixel_f"+frmidx+"c"+camidx+".png", SPContour);

		delete pSP;
		//namedWindow("Superpixel "+frmidx, 0);
		//imshow("Superpixel "+frmidx, SPContour); //waitKey();
	}
	else
	{
		int lcnt;
		spIFile >> lcnt;
		mvd->lCntVec.push_back(lcnt);

		cv::Point p;
		Mat spMap2(img8.rows,img8.cols,CV_32FC1);
		for(p.y=0; p.y<img8.rows; p.y++)
			for(p.x=0; p.x<img8.cols; p.x++)
				spIFile >> spMap2.at<float>(p);
		mvd->dsSpVec.push_back(spMap2);

		Mat spClMat;
		Mat spClMat8 = imread(dir+"sp_f"+frmidx+"c"+camidx+".png");
		spClMat8.convertTo(spClMat,CV_64FC3);
		mvd->spClVec.push_back(spClMat);
		mvd->tolLabel += lcnt;

		Mat SPContour = imread(dir+"Superpixel_f"+frmidx+"c"+camidx+".png");
		//namedWindow("Superpixel "+frmidx, 0);
		//imshow("Superpixel "+frmidx, SPContour); //waitKey();
		spIFile.close();
	}
		
}



void on_mouseclick( int event, int x, int y, int flags, void* param )
{
    gcui.mouseClick( event, x, y, flags, param );
}

void calcSIFTFlow(Mat& Im1, Mat& Im2, Mat& flow)
{

	DImage imd1, imd2;
	imd1.loadCVImage(Im1);
	imd2.loadCVImage(Im2);
	//bool IsMultiScale = true;
	int cellSize[] = {3};
	vector<int> cellSizeVect(cellSize,cellSize+sizeof(cellSize)/sizeof(int));
	int stepSize = 1;
	bool IsBoundaryIncluded = true;
		
	UCImage imsift1, imsift2;
	ImageFeature::imSIFT(imd1,imsift1,cellSizeVect,stepSize,IsBoundaryIncluded);
	ImageFeature::imSIFT(imd2,imsift2,cellSizeVect,stepSize,IsBoundaryIncluded);

	imsift1.GaussianSmoothing(imsift1,.8,5);
	imsift2.GaussianSmoothing(imsift2,.8,5);

	double alpha = 2*255;//2*255;
	double d = 20*alpha;
	double gamma = 0.005*255;
	BPFlow bpflow;
	int wsize = 5; //5
    int nlevels = 4; //2
	int nIterations = 60;

	
	bpflow.setDataTermTruncation(true);
	//bpflow.setTRW(true);
	//bpflow.setDisplay(false);
	bpflow.LoadImages(imsift1.width(),imsift1.height(),imsift1.nchannels(),imsift1.data(),imsift2.data());
	bpflow.setPara(alpha,d);
	//bpflow.setPara(imsift1,imsift2);
	bpflow.setHomogeneousMRF(wsize);
	bpflow.ComputeDataTerm();
	bpflow.ComputeRangeTerm(gamma);
	bpflow.MessagePassing(nIterations,nlevels);
	bpflow.ComputeVelocity();

	flow.create(Im1.rows,Im1.cols,CV_32FC2);
	
	for(int i=0;i<imsift1.height();i++)
	{
		for(int j=0;j<imsift1.width();j++)
		{
			flow.at<cv::Point2f>(i,j).x = bpflow.flow().data()[(i*Im1.cols+j)*2]; 
			flow.at<cv::Point2f>(i,j).y = bpflow.flow().data()[(i*Im1.cols+j)*2+1];
		}
	}


}