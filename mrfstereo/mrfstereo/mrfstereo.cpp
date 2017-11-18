#include "mrfstereo.h"


void computeDSI(CByteImage im1,       // source (reference) image
		CByteImage im2,       // destination (match) image
		MRF::CostVal *&dsi,   // computed cost volume
		int nD,               // number of disparities
		int birchfield,       // use Birchfield/Tomasi costs
		int squaredDiffs,     // use squared differences
		int truncDiffs)       // truncated differences
{
    CShape sh = im1.Shape();
    int width = sh.width, height = sh.height, nB = sh.nBands;
    dsi = new MRF::CostVal[width * height * nD];

    int nColors = __min(3, nB);

    // worst value for sumdiff below 
    int worst_match = nColors * (squaredDiffs ? 255 * 255 : 255);
    // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
    int maxsumdiff = squaredDiffs ? truncDiffs * truncDiffs : nColors * abs(truncDiffs);
    // value for out-of-bounds matches
    int badcost = __min(worst_match, maxsumdiff);

    int dsiIndex = 0;
    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    uchar *pix1 = &im1.Pixel(x, y, 0);
	    for (int d = 0; d < nD; d++) {
		int x2 = x-d;
		int dsiValue;
                if (x2 >= 0 && d < nD) { // in bounds
		    uchar *pix2 = &im2.Pixel(x2, y, 0);
                    int sumdiff = 0;
                    for (int b = 0; b < nColors; b++) {
			int diff = 0;
			if (birchfield) {
			    // Birchfield/Tomasi cost
			    int im1c = pix1[b];
			    int im1l = x == 0?   im1c : (im1c + pix1[b - nB]) / 2;
			    int im1r = x == width-1? im1c : (im1c + pix1[b + nB]) / 2;
			    int im2c = pix2[b];
			    int im2l = x2 == 0?   im2c : (im2c + pix2[b - nB]) / 2;
			    int im2r = x2 == width-1? im2c : (im2c + pix2[b + nB]) / 2;
			    int min1 = __min(im1c, __min(im1l, im1r));
			    int max1 = __max(im1c, __max(im1l, im1r));
			    int min2 = __min(im2c, __min(im2l, im2r));
			    int max2 = __max(im2c, __max(im2l, im2r));
			    int di1 = __max(0, __max(im1c - max2, min2 - im1c));
			    int di2 = __max(0, __max(im2c - max1, min1 - im2c));
			    diff = __min(di1, di2);
			} else {
			    // simple absolute difference
			    int di = pix1[b] - pix2[b];
			    diff = abs(di);
			}
			// square diffs if requested (Birchfield too...)
			sumdiff += (squaredDiffs ? diff * diff : diff);
                    }
		    // truncate diffs
		    dsiValue = __min(sumdiff, maxsumdiff);
                } else { // out of bounds: use maximum truncated cost
		    dsiValue = badcost;
		}
		//int x0=-140, y0=-150;
		//if (x==x0 && y==y0)
		//    printf("dsi(%d,%d,%2d)=%3d\n", x, y, d, dsiValue); 

		// The cost of pixel p and label l is stored at dsi[p*nLabels+l]
		dsi[dsiIndex++] = dsiValue;
	    }
	}
    }
    //exit(1);
}

void computeCues(CByteImage im1, MRF::CostVal *&hCue, MRF::CostVal *&vCue,
		 int gradThresh, int gradPenalty) {
    CShape sh = im1.Shape();
    int width = sh.width, height = sh.height, nB = sh.nBands;
    hCue = new MRF::CostVal[width * height];
    vCue = new MRF::CostVal[width * height];

    int nColors = __min(3, nB);

    // below we compute sum of squared colordiffs, so need to adjust threshold accordingly (results in RMS)
    gradThresh *= nColors * gradThresh;

    //sh.nBands=1;
    //CByteImage hc(sh), vc(sh);
    int n = 0;
    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    uchar *pix   = &im1.Pixel(x, y, 0);
	    uchar *pix1x = &im1.Pixel(x + (x < width-1), y, 0);
	    uchar *pix1y = &im1.Pixel(x, y + (y < height-1), 0);
	    int sx = 0, sy = 0;
	    for (int b = 0; b < nColors; b++) {
		int dx = pix[b] - pix1x[b];
		int dy = pix[b] - pix1y[b];
		sx += dx * dx;
		sy += dy * dy;
	    }
	    hCue[n] = (sx < gradThresh ? gradPenalty : 1);
	    vCue[n] = (sy < gradThresh ? gradPenalty : 1);
	    //hc.Pixel(x, y, 0) = 100*hCue[n];
	    //vc.Pixel(x, y, 0) = 100*vCue[n];
	    n++;
	}
    }
    //WriteImageVerb(hc, "hcue.png", true);
    //WriteImageVerb(vc, "vcue.png", true);
    //exit(1);
}

void WTA(MRF::CostVal *dsi, int width, int height, int nD, CByteImage &disp)
{
    CShape sh(width, height, 1);
    disp.ReAllocate(sh);
    int n = 0;
    for (int y = 0; y < height; y++) {
	uchar *row = &disp.Pixel(0, y, 0);
	for (int x = 0; x < width; x++) {
	    int minval = dsi[n++]; // dsi(x,y,0)
	    int mind = 0;
	    for (int d = 1; d < nD; d++) {
		int val = dsi[n++]; // dsi(x,y,d)
		if (val < minval) {
		    minval = val;
		    mind = d;
		}
	    }
	    row[x] = mind;
	}
    }
}

void getDisparities(MRF *mrf, int width, int height, CByteImage &disp)
{
    CShape sh(width, height, 1);
    disp.ReAllocate(sh);

    int n = 0;
    for (int y = 0; y < height; y++) {
	uchar *row = &disp.Pixel(0, y, 0);
	for (int x = 0; x < width; x++) {
	    row[x] = mrf->getLabel(n++);
	}
    }
}

void setDisparities(CByteImage disp, MRF *mrf)
{
    CShape sh = disp.Shape();
    int width = sh.width, height = sh.height;

    int n = 0;
    for (int y = 0; y < height; y++) {
	uchar *row = &disp.Pixel(0, y, 0);
	for (int x = 0; x < width; x++) {
	    mrf->setLabel(n++, row[x]);
	}
    }
}

void writeDisparities(FILE* debugfile, CByteImage disp, int outscale, char *dispname, int verbose, int writeParams)
{
    if (verbose || writeParams)
	fprintf(debugfile, "scaling disparities by %d\n", outscale);
    CByteImage disp2;
    ScaleAndOffset(disp, disp2, (float)outscale, 0);
    
    //WriteImageVerb(disp2, dispname, verbose);
}