

#include "CTexton.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "dirop.h"
#include <vector>
#include <fstream>



using namespace std;
using namespace cv;

typedef cv::Vec<float,40> VecTexton;

CTexton::CTexton(Mat& img, int vocabSize):m_vocabSize(vocabSize)
{
	m_img = img.clone();
	makeRFSfilters();
	featureFilters();

	vector<Mat> allFeats = m_fullFeats;
	Mat featMat64(m_img.rows,m_img.cols,CV_64FC(40));
	cv::merge(allFeats,featMat64);
	featMat64.convertTo(m_featMat,CV_32FC(40));
}
///*
CTexton::CTexton(Mat& img, Mat& vocabulary, int vocabSize):m_vocabSize(vocabSize)
{
	m_img = img.clone();
	m_vocabulary = vocabulary.clone();
	makeRFSfilters();
	featureFilters();

	vector<Mat> allFeats = m_fullFeats;
	Mat featMat64(m_img.rows,m_img.cols,CV_64FC(40));
	cv::merge(allFeats,featMat64);
	featMat64.convertTo(m_featMat,CV_32FC(40));
}//*/

CTexton::~CTexton(void)
{
}

void CTexton::buildVocab(Mat& vocabulary, Mat& label, string& dir)
{

	string filename = makeVocabularyFileName(dir+"Texton", m_vocabSize);

	///*
	ifstream iop;
	iop.open(filename);
	if(iop.is_open())
	{
		m_vocabulary = Mat(m_vocabSize,40,CV_32FC1);
		m_cLabels = Mat(m_img.rows*m_img.cols,1,CV_32FC1);

		for(int i=0; i<m_vocabSize; i++)
			for(int j=0; j<40; j++)
				iop >> m_vocabulary.at<float>(i,j);

		for(int i=0; i<m_img.rows*m_img.cols; i++)
		{
			iop >> m_cLabels.at<int>(i,0);
			//cout << "m_cLabels.at<int>(i,0) = " << m_cLabels.at<int>(i,0) <<endl;
		}

		iop.close();
		//label = m_cLabels.clone();
		//return;
	}//*/
	else
	{
		iop.close();
		const int kMeansItCount = 50;
		const int kMeansType = KMEANS_PP_CENTERS;

		///*
		Mat cSamples( m_img.rows*m_img.cols, 40, CV_32FC1);
		for(int i=0; i<m_img.rows; i++)
		{
			for(int j=0; j<m_img.cols; j++)
			{
				VecTexton desc = m_featMat.at<VecTexton>(i,j);

				for(int k=0; k<40; k++)
				{
					cSamples.at<float>(i*m_img.cols+j,k) = desc[k];
				}
			}
		}

		kmeans( cSamples, m_vocabSize, m_cLabels,
			TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, m_vocabulary );

		ofstream fop;
		fop.open(filename);
		for(int i=0; i<m_vocabSize; i++)
		{
			for(int j=0; j<40; j++)
			{
				fop << (float) m_vocabulary.at<float>(i,j);
				//cout << "m_vocabulary.at<float>(i,j) = " << m_vocabulary.at<float>(i,j) << endl;
				fop << " ";
			}
			fop << endl;
		}
		for(int i=0; i<m_img.rows*m_img.cols; i++)
		{
			fop << (float) m_cLabels.at<int>(i,0);
			fop << " ";
		}

		fop.close();
	}

	//label = Mat(m_img.rows, m_img.cols,CV_32FC1,m_cLabels.data);
	vocabulary = m_vocabulary.clone();
	label = Mat(m_img.rows, m_img.cols,CV_32FC1);
	for(int i=0; i<m_img.rows; i++)
		for(int j=0; j<m_img.cols; j++)
			label.at<float>(i,j) = m_cLabels.at<int>(i*m_img.cols+j,0);


	//imshow("textonmap", label/(double)(m_vocabSize-1)*255);
	//waitKey();
/*
	m_textonColorMap = Mat(m_img.rows,m_img.cols,CV_8UC3);
	Vector<Scalar> textonColor(m_vocabSize, Scalar::all(-1));

	for(int i=0; i<m_img.rows; i++)
	{
		for(int j=0; j<m_img.cols; j++)
		{
			if(textonColor[label.at<float>(i,j)]==Scalar::all(-1))
			{
				RNG& rng = theRNG();
				Scalar color(rng(256), rng(256), rng(256));
				textonColor[label.at<float>(i,j)] = color;
			}
			m_textonColorMap.at<Vec3b>(i,j)[0] = textonColor[label.at<float>(i,j)].val[0];
			m_textonColorMap.at<Vec3b>(i,j)[1] = textonColor[label.at<float>(i,j)].val[1];
			m_textonColorMap.at<Vec3b>(i,j)[2] = textonColor[label.at<float>(i,j)].val[2];
		}
	}*/
	//imshow("Textoncolormap", m_textonColorMap);
	//waitKey();

}

void CTexton::extractTexton(Mat& textonMap)
{
	/*
	string filename = makeVocabularyFileName("./temp.png", m_vocabSize);
	ifstream iop;
	iop.open(filename);
	if(iop.is_open())
	{
		for(int i=0; i<m_vocabSize; i++)
			for(int j=0; j<40; j++)
				iop >> m_vocabulary.at<float>(i,j);
		//return m_vocabulary;
	}
	iop.close();*/

	m_textonMap = Mat(m_img.rows,m_img.cols,CV_32FC1);
	m_textonColorMap = Mat(m_img.rows,m_img.cols,CV_8UC3);
	Vector<Scalar> textonColor(m_vocabSize, Scalar::all(-1));

	for(int i=0; i<m_featMat.rows; i++)
	{
		for(int j=0; j<m_featMat.cols; j++)
		{
			VecTexton desc = m_featMat.at<VecTexton>(i,j);
			/*cout << "desc[0] = " << desc[0] << endl;
			cout << "desc[1] = " << desc[1] << endl;
			cout << "desc[2] = " << desc[2] << endl;
			cout << "desc[3] = " << desc[3] << endl;*/
			double minDescDist = 10000000;
			double minVocId;
			for(int l=0; l<m_vocabulary.rows; l++)
			{
				float* pvoc = m_vocabulary.ptr<float>(l);
				double descDist = 0.0;
				for(int k=0; k<40; k++)
				{
					 //descDist += pow(desc[k]-pvoc[k],2.0);
					descDist += abs(desc[k]-pvoc[k]);
				}
				//descDist = sqrt(descDist);
				if(descDist<minDescDist)
				{
					minVocId = l;
					minDescDist = descDist;
				}
			}
			//cout << "minVocId = " << minVocId << endl;
			m_textonMap.at<float>(i,j) = minVocId;
			if(textonColor[minVocId]==Scalar::all(-1))
			{
				RNG& rng = theRNG();
				Scalar color(rng(256), rng(256), rng(256));
				textonColor[minVocId] = color;
			}
			m_textonColorMap.at<Vec3b>(i,j)[0] = textonColor[minVocId].val[0];
			m_textonColorMap.at<Vec3b>(i,j)[1] = textonColor[minVocId].val[1];
			m_textonColorMap.at<Vec3b>(i,j)[2] = textonColor[minVocId].val[2];
		}
	}
	//imshow("Textoncolormap", m_textonColorMap);

	textonMap = m_textonMap.clone();
	//waitKey();
	//return m_textonMap;
}


void CTexton::extractTextonEM(Mat& textonMap,cv::EM& em_model)
{

	m_textonMap = Mat(m_img.rows,m_img.cols,CV_32FC1);
	m_textonColorMap = Mat(m_img.rows,m_img.cols,CV_8UC3);
	Vector<Scalar> textonColor(m_vocabSize, Scalar::all(-1));

	for(int i=0; i<m_featMat.rows; i++)
	{
		for(int j=0; j<m_featMat.cols; j++)
		{
			VecTexton desc = m_featMat.at<VecTexton>(i,j);
			Mat sample(desc);

			Vec2d em_result;
			em_result = em_model.predict(sample);
			
			m_textonMap.at<float>(i,j) = em_result[1];
			if(textonColor[em_result[1]]==Scalar::all(-1))
			{
				RNG& rng = theRNG();
				Scalar color(rng(256), rng(256), rng(256));
				textonColor[em_result[1]] = color;
			}
			m_textonColorMap.at<Vec3b>(i,j)[0] = textonColor[em_result[1]].val[0];
			m_textonColorMap.at<Vec3b>(i,j)[1] = textonColor[em_result[1]].val[1];
			m_textonColorMap.at<Vec3b>(i,j)[2] = textonColor[em_result[1]].val[2];
		}
	}
	//imshow("Textoncolormap", m_textonColorMap);

	textonMap = m_textonMap.clone();
	//waitKey();
	//return m_textonMap;
}

void CTexton::buildVocabEM(Mat& vocabulary, Mat& label)
{

	//string filename = makeVocabularyFileName("./TextonEM", m_vocabSize);
	EM em_model(m_img.rows*m_img.cols);
	string emfilename("em_model.xml");
	FileStorage fs(emfilename, FileStorage::READ);


	string filename = makeVocabularyFileName("./TextonEM", m_vocabSize);
	ifstream iop;
	iop.open(filename);

	if(fs.isOpened() && iop.is_open())
	{
		em_model.read(fs["em_model"]);
		fs.release();

		m_cLabels = Mat(m_img.rows*m_img.cols,1,CV_32FC1);

		for(int i=0; i<m_img.rows*m_img.cols; i++)
		{
			iop >> m_cLabels.at<int>(i,0);
		}

		iop.close();

	}//*/
	else
	{

		///*
		Mat cSamples( m_img.rows*m_img.cols, 40, CV_32FC1);
		for(int i=0; i<m_img.rows; i++)
		{
			for(int j=0; j<m_img.cols; j++)
			{
				VecTexton desc = m_featMat.at<VecTexton>(i,j);

				for(int k=0; k<40; k++)
				{
					cSamples.at<float>(i*m_img.cols+j,k) = desc[k];
				}
			}
		}


		Mat labels( m_img.rows*m_img.cols, 1, CV_32SC1 );

		// cluster the data
		em_model.train( cSamples, noArray(), labels );

		m_cLabels = labels.clone();

		WriteStructContext ws(fs, "em_model", CV_NODE_MAP);
		em_model.write(fs);

		ofstream fop;
		fop.open(filename);
		for(int i=0; i<m_img.rows*m_img.cols; i++)
		{
			fop << (float) m_cLabels.at<int>(i,0);
			fop << " ";
		}

		fop.close();
	}

	label = Mat(m_img.rows, m_img.cols,CV_32FC1);
	for(int i=0; i<m_img.rows; i++)
		for(int j=0; j<m_img.cols; j++)
			label.at<float>(i,j) = m_cLabels.at<int>(i*m_img.cols+j,0);


	//imshow("textonmap", label/(double)(m_vocabSize-1)*255);
	//waitKey();

	m_textonColorMap = Mat(m_img.rows,m_img.cols,CV_8UC3);
	Vector<Scalar> textonColor(m_vocabSize, Scalar::all(-1));

	for(int i=0; i<m_img.rows; i++)
	{
		for(int j=0; j<m_img.cols; j++)
		{
			if(textonColor[label.at<float>(i,j)]==Scalar::all(-1))
			{
				RNG& rng = theRNG();
				Scalar color(rng(256), rng(256), rng(256));
				textonColor[label.at<float>(i,j)] = color;
			}
			m_textonColorMap.at<Vec3b>(i,j)[0] = textonColor[label.at<float>(i,j)].val[0];
			m_textonColorMap.at<Vec3b>(i,j)[1] = textonColor[label.at<float>(i,j)].val[1];
			m_textonColorMap.at<Vec3b>(i,j)[2] = textonColor[label.at<float>(i,j)].val[2];
		}
	}
	imshow("Textoncolormap", m_textonColorMap);
	waitKey();

}

void CTexton::featureFilters()
{

	//cout << "OK" << endl;
	
	Mat luv_im, luv32;
	Mat im32;
	m_img.convertTo(im32,CV_32FC3);
	im32 *= 1./255;
	cvtColor(im32, luv32, CV_BGR2Luv);
	luv32.convertTo(luv_im,CV_64FC3);

	vector<Mat> luv_planes;
	luv_planes.resize(3);
	split( luv_im, luv_planes );
  
	Mat L = luv_planes[0];
	Mat u = luv_planes[1];
	Mat v = luv_planes[2];

	//imwrite("./L.png", L/255.0);
	
	int delta = 0;
	int ddepth = -1;
	Mat dst;
	Mat kernel;
	cv::Point anchor = Point( -1, -1 );
	
	vector<Mat> L_edge_responses;
	//L_edge_responses.resize(18);
	vector<Mat> L_bar_responses;
	//L_bar_responses.resize(18);

	Mat L_log_response;
	Mat L_gaussian_response;
	Mat u_gaussian_response;
	Mat v_gaussian_response;
	
	for(int j=0; j<18; j++)
	{
	      kernel = m_F[j];
	      filter2D( L, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT );
		  L_edge_responses.push_back(dst.clone());
	      //L_edge_responses[j] = dst;
		  //imshow("L_edge",L_edge_responses[j]);
		  //waitKey();
	}
	

	for(int j=18; j<36; j++)
	{
	      kernel = m_F[j];
	      filter2D( L, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT );
	      L_bar_responses.push_back(dst.clone());
		  //imshow("L_bar",L_bar_responses[j-18]);
		  //waitKey();
	}
	
	filter2D( L, L_log_response, ddepth, m_F[37], anchor, delta, BORDER_DEFAULT );
	//imshow("L_log_response",L_log_response);
	filter2D( L, L_gaussian_response, ddepth, m_F[36], anchor, delta, BORDER_DEFAULT );
	//imshow("L_gaussian_response",L_gaussian_response);
	filter2D( u, u_gaussian_response, ddepth, m_F[36], anchor, delta, BORDER_DEFAULT );
	//imshow("u_gaussian_response",u_gaussian_response);
	filter2D( v, v_gaussian_response, ddepth, m_F[36], anchor, delta, BORDER_DEFAULT );
	//imshow("v_gaussian_response",v_gaussian_response);
	//waitKey();
	
	vector<Mat> L_max_edge_responses;
	vector<Mat> L_max_bar_responses;
	
	int startind = 0;
	for(int scale=0; scale<3; scale++)
	{
	      //vector<Mat> edge_responses_this_scale;
	      //vector<Mat> bar_responses_this_scale;

		  Mat maxEdgeMat = L_edge_responses[startind].clone();
	      Mat maxBarMat = L_bar_responses[startind].clone();
		  for(int l=1; l<6; l++)
		  {
				//imshow("L_edge_responses[startind+l]",L_edge_responses[startind+l]);
				//imshow("L_bar_responses[startind+l]",L_bar_responses[startind+l]);
				//waitKey();

				cv::max(maxEdgeMat,L_edge_responses[startind+l],maxEdgeMat);
				cv::max(maxBarMat,L_bar_responses[startind+l],maxBarMat);
		  }

		  //imshow("maxEdgeMat",maxEdgeMat);
		  //imshow("maxBarMat",maxBarMat);
		  //waitKey();
	      L_max_edge_responses.push_back(maxEdgeMat);
	      L_max_bar_responses.push_back(maxBarMat);
		  startind += 6;
	}
	
	//vector<Mat> fullFeats;
	
	m_fullFeats.push_back(L_log_response.clone());
	m_fullFeats.push_back(L_gaussian_response.clone());
	m_fullFeats.push_back(u_gaussian_response.clone());
	m_fullFeats.push_back(v_gaussian_response.clone());
	
	for(int i=0; i<18; i++)
	      m_fullFeats.push_back(L_edge_responses[i].clone());
	
	for(int i=0; i<18; i++)
	      m_fullFeats.push_back(L_bar_responses[i].clone());
	
	
	m_rotInvarFeats.assign(m_fullFeats.begin(), m_fullFeats.begin()+4);
	for(int scale=0; scale<3; scale++)
	      m_rotInvarFeats.push_back(L_max_edge_responses[scale].clone());
	for(int scale=0; scale<3; scale++)
	      m_rotInvarFeats.push_back(L_max_bar_responses[scale].clone());

	//cout << "All feature extracted" << endl;
	
 }

void CTexton::gauss1d(double sigma, double mean, double x, int ord, double & g)
{
	x = x-mean;
	double num = x*x;
	double variance = sigma*sigma;
	double denom = 2*variance;
	g = (double)exp(-num/denom)/(double)sqrt(CV_PI*denom);
	
	switch(ord)
	{
	      case 1: 
		      g=-(double)g*((double)x/(double)variance);
		      break;
	      case 2: 
		      g=(double)g*((double)(num-variance)/(double)(variance*variance));
			  //cout << "g = " << g << endl;
		      break; 
	}
  
}


void CTexton::makefilter(double scale, int phasex, int phasey, Mat& pts, int SUP, Mat& f)
{
  	f = Mat(SUP,SUP,CV_64FC1).clone();
	double sum = 0;
	cv::Point p;
	for(p.x=0; p.x<SUP*SUP; p.x++)
	{
		int dx = floor((double)p.x/(double)SUP);
		int dy = p.x-dx*SUP;
		double gx = 0, gy = 0;
		for(p.y=0; p.y<2; p.y++)
		{
			if(p.y==0)
				gauss1d(3*scale,0,pts.at<double>(p),phasex,gx);
			else
				gauss1d(scale,0,pts.at<double>(p),phasey,gy);
		}
		      
		f.at<double>(dy,dx) = gx*gy;
		sum += gx*gy;
	}

	
	//!< normalise
	double mean = (double)sum/(double)(SUP*SUP);
	//cout << "mean = " << mean << endl;
	f = f-mean;
	double abssum = 0;
	///*
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for(p.x=0; p.x<SUP; p.x++)
	      {
			  abssum += (double)abs(f.at<double>(p));
			  //cout << "f = " << f.at<double>(p) << endl;
		  }
		  //cout << endl;
	}
	//cout << endl;//*/
	
	//Scalar abssum_scalar = cv::sum(cv::abs(f));
	//abssum = abssum_scalar.val[0];
	//cout << "abssum = " << abssum<< endl;
	
	if(abssum>0)
		f = f/(double)abssum;
	///*

	sum = 0;
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for(p.x=0; p.x<SUP; p.x++)
	      {
			  //cout << "f = " << f.at<double>(p) << endl;
			  sum += (double)f.at<double>(p);
		  }
		  //cout << endl;
	}
	//cout << endl;

	//Scalar sum_scalar = cv::sum(f);
	//sum = sum_scalar.val[0];
	//cout << "sum = " << sum << endl;

	/*
	double minV, maxV;
	cv::minMaxLoc(f,&minV,&maxV);
	cout << "minV = " << minV << endl;
	cout << "maxV = " << maxV << endl;*/

}

void CTexton::gaussianFilter(Mat& H, int size, double std)
{
  
	H = Mat(size,size,CV_64FC1).clone();
  	int hsup = (size-1)/2;
	//int disp = hsup+1;
	
	cv::Point p;
	double sum = 0;
	//Mat meshgridMat(size,size,CV_64FC2);
	for(p.y=0; p.y<size; p.y++)
	{
	      for(p.x=0; p.x<size; p.x++)
	      {
		      double x = p.x-hsup;
		      double y = p.y-hsup;
		      double h = exp(-(x*x + y*y)/(2*std*std));
		      H.at<double>(p) = h;
		      sum += h;     
	      }
	}
	
	if(sum!=0)
	      H = H/sum;

	//!< normalise
	Scalar sum_scalar = cv::sum(H);
	sum = sum_scalar.val[0];
	double mean = sum/(double)(size*size);
	H = H-mean;
	
	Scalar abssum_scalar = cv::sum(cv::abs(H));
	double abssum = abssum_scalar.val[0];
	
	if(abssum>0)	
		H = H/abssum;
}

void CTexton::lapGaussianFilter(Mat& H, int size, double std)
{
  
  	H = Mat(size,size,CV_64FC1).clone();
  	int hsup = (size-1)/2;
	//int disp = hsup+1;
	double std2 = std*std;
	
	cv::Point p;
	double sum = 0;
	//Mat meshgridMat(size,size,CV_64FC2);
	for(p.y=0; p.y<size; p.y++)
	{
	      for(p.x=0; p.x<size; p.x++)
	      {
		      double x = p.x-hsup;
		      double y = p.y-hsup;
		      double h = exp(-(x*x + y*y)/(2*std*std));
		      H.at<double>(p) = h;
		      sum += h;     
	      }
	}
	//cout << "H=" << H.at<float>(0,0) << endl;
	
	if(sum!=0)
	      H = H/sum;

	sum = 0;
	for(p.y=0; p.y<size; p.y++)
	{
	      for(p.x=0; p.x<size; p.x++)
	      {
		      double x = p.x-hsup;
		      double y = p.y-hsup;
			  //cout << "H=" << H.at<float>(p) << endl;
		      double h = H.at<double>(p) * ((double)x*x + (double)y*y-2*std2)/(double)(std2*std2);
		      sum += h;     
		      
		      H.at<double>(p) = h;
	      }
	}

	H = H - sum/(double)(size*size);

	//!< normalise
	Scalar sum_scalar = cv::sum(H);
	sum = sum_scalar.val[0];

	double mean = sum/(double)(size*size);
	H = H-mean;
	
	Scalar abssum_scalar = cv::sum(cv::abs(H));
	double abssum = abssum_scalar.val[0];
	//cout << "abssum=" << abssum << endl;
	
	if(abssum>0)	
		H = H/abssum;

	//imshow("log",H*255);
	//waitKey();	
	
}

void CTexton::makeRFSfilters()
{

	///TODO: extend it to: 8 LoG, 4 Gaussians, scale from 1 to 10 (the Leung-Malik (LM) set)
	///refer to http://www.robots.ox.ac.uk/~vgg/research/texclass/papers/varma05.pdf for details

	int SUP = 49; 		//!< Support of the largest filter (must be odd)
	double SCALEXARR[] = {1,2,4}; 	//!< Sigma_{x} for the oriented filters
	//double SCALEXARR[] = {4,8,16};
	vector<double> SCALEX;
	SCALEX.assign(SCALEXARR,SCALEXARR+3);
	
	int NORIENT = 6;              	//!< Number of orientations
	int NROTINV = 2;
	int NBAR = SCALEX.size()*NORIENT;
	int NEDGE = SCALEX.size()*NORIENT;
	int NF = NBAR+NEDGE+NROTINV;
	
	//Mat F(SUP,SUP,CV_64FC(NF);
	//vector<Mat> F;
	m_F.resize(NF);
	
	int hsup = (SUP-1)/2;
	//int disp = hsup+1;
	
	cv::Point p;
	Mat meshgridMat(SUP,SUP,CV_64FC2);
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for( p.x=0; p.x<SUP; p.x++)
	      {
		      meshgridMat.at<Vec2d>(p)[0] = p.x-hsup;
			  meshgridMat.at<Vec2d>(p)[1] = p.y-hsup;
		      //meshgridMat.at<Vec2d>(p)[1] = hsup-p.y;
	      }
	}
	
	Mat orgPts(2,SUP*SUP,CV_64FC1);
	for(p.y=0; p.y<2; p.y++)
	{
	      for(p.x=0; p.x<SUP*SUP; p.x++)
	      {
		      int dx = floor((float)p.x/SUP);
		      int dy = p.x-dx*SUP;
		      orgPts.at<double>(p) = meshgridMat.at<Vec2d>(dy,dx)[p.y];
	      }
	}  
	
	//cout << "OK0" << endl;

	int count = 0;
	for(int scale=0; scale<SCALEX.size(); scale++)
	{
	      for(int orient=0; orient<NORIENT; orient++)
	      {
		      double angle = CV_PI*(double)orient/(double)NORIENT; //!< Not 2pi as filters have symmetry
			  //cout << "orient = " << orient << endl;
			  //cout << "angle = " << angle << endl;
		      double c = cos(angle);
		      double s = sin(angle);
		      Mat rotMat = (Mat_<double>(2,2) << c, -s, s, c);
		      Mat rotPts = rotMat*orgPts;

		      Mat f;
		      makefilter(SCALEX[scale],0,1,rotPts,SUP,f);
		      m_F[count] = f;
			  /*
			  for(p.y=0; p.y<SUP; p.y++)
			  {
					for(p.x=0; p.x<SUP; p.x++)
					{
						cout << "f = " << F[count].at<double>(p) << endl;
					}
					cout << endl;
			  }
			  cout << endl;*/

			  //Scalar sum_scalar = cv::sum(F[count]);
			  //double sum = sum_scalar.val[0];
		      //cout << "sum=" << sum << endl;
			  //imshow("f",f*255);
			  //waitKey();
		      makefilter(SCALEX[scale],0,2,rotPts,SUP,f);
		      m_F[count+NEDGE] = f;
			  /*
			  for(p.y=0; p.y<SUP; p.y++)
			  {
					for(p.x=0; p.x<SUP; p.x++)
					{
						cout << "f = " << F[count+NEDGE].at<double>(p) << endl;
					}
					cout << endl;
			  }
			  cout << endl;*/
			  //sum_scalar = cv::sum(F[count+NEDGE]);
			  //sum = sum_scalar.val[0];
		      //cout << "sum=" << sum << endl;
			  //imshow("f",f*255);
			  //waitKey();
		      count++;
	      }
	}
	
	//cout << "OK1" << endl;
	
	Mat gaussF;
	gaussianFilter(gaussF, SUP, 10);
	m_F[NBAR+NEDGE] = gaussF;
	//imshow("gaussF",gaussF*255);
	//waitKey();
	
	Mat lGaussF;
	lapGaussianFilter(lGaussF, SUP, 10);
	//cout << "OK3" << endl;
	m_F[NBAR+NEDGE+1] = lGaussF;
	//imshow("lGaussF",lGaussF*255);
	//waitKey();
}