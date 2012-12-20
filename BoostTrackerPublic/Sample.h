// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_SAMPLE
#define H_SAMPLE

#include "Matrix.h"
#include "Public.h"


/******************************* Particle Properity Definitions *********************************/

/* standard deviations for gaussian sampling in transition model */
//#define TRANS_X_STD 1.0
//#define TRANS_Y_STD 0.5

#define TRANS_X_STD 6.0
#define TRANS_Y_STD 6.0
#define TRANS_S_STD 0.2
#define PARTICLENUM 800
#define UPDATEFREQUENCE 2

/* autoregressive dynamics parameters for transition model */
#define A1  2.0
#define A2 -1.0
#define B0  1.5

class Sample;


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class Sample
{
public:
						Sample(Matrixu *img, int row, int col, int width=0, int height=0, float weight=1.0);
						Sample() { _img = NULL; _row = _col = _height = _width = 0; _weight = 1.0f;};
	Sample&				operator= ( const Sample &a );

	/////////////////////////
	void transition(Matrixu *img);
	//static void setImageSize(int width, int height){ imageWidth = width; imageHeight = height;};
	// void setImageSize(int width, int height){ imageWidth = width; imageHeight = height;};

	// used in sort method so that the vector can 升序？？ 降序？？？ 我估计是降序
	bool operator < (const Sample& s1 ) const ;
public:
	Matrixu				*_img;
	int					_row, _col, _width, _height;
	float				_weight;
	// new added //
	float scale;          /**< scale */
    float xPre;         /**< previous x coordinate */
    float yPre;         /**< previous y coordinate */
    float sPre;         /**< previous scale */
    float x0;         /**< original x coordinate */
    float y0;         /**< original y coordinate */
    int originWidth;        /**< original width of region described by particle */
    int originHeight;       /**< original height of region described by particle */	
	//static int imageWidth;
	//static int imageHeight;
   	float XmaxBoundary;
	float YmaxBoundary;
};

//int Sample::imageHeight;
//int Sample::imageWidth;


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class SampleSet
{
public:
						SampleSet() {};
						SampleSet(const Sample &s) { _samples.push_back(s); };

	int					size() const { return _samples.size(); };
	void				push_back(const Sample &s) { _samples.push_back(s); };
	void				push_back(Matrixu *img, int x, int y, int width=0, int height=0, float weight=1.0f);
	void				resize(int i) { _samples.resize(i); };
	void				resizeFtrs(int i);
	float &				getFtrVal(int sample,int ftr) { return _ftrVals[ftr](sample); };
	float				getFtrVal(int sample,int ftr) const { return _ftrVals[ftr](sample); };
	Sample &			operator[] (const int sample)  { return _samples[sample]; };
	Sample				operator[] (const int sample) const { return _samples[sample]; };
	Matrixf				ftrVals(int ftr) const { return _ftrVals[ftr]; };
	bool				ftrsComputed() const { return !_ftrVals.empty() && !_samples.empty() && _ftrVals[0].size()>0; };
	void				clear() { _ftrVals.clear(); _samples.clear(); };

	
						// densly sample the image in a donut shaped region: will take points inside circle of radius inrad,
						// but outside of the circle of radius outrad.  when outrad=0 (default), then just samples points inside a circle
	void				sampleImage(Matrixu *img, int x, int y, int w, int h, float inrad, float outrad=0, int maxnum=1000000);
	void				sampleImage(Matrixu *img, uint num, int w, int h);

	/////////////////////////////////////////////////

	void init_particle_distributions(Matrixu *img,int x, int y, int w, int h, int maxnum=100);


	// here we use the second ... method to let the particle move like Borown model
	void sampleParticles(Matrixu *img);
	void normalize_weights();

   /* use the standard 残差 method */
    void resample_1();
	
	/* also above method, but different implementation */
	void resample_2();
	bool compare_weight( const Sample* s1, const Sample* s2){ return s1->_weight < s2->_weight; };

	void weightParticles( vectorf weights);

	// find the best particle
	 Sample findBestParticle();
     
	 //
	 Sample findMeanParticle();

private:
	vector<Sample>		_samples;
	vector<Matrixf>		_ftrVals; // [ftr][sample]
	vector<Sample>    newSamples;   // sample pointers after resample

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


// 记住构造函数啊
inline Sample&			Sample::operator= ( const Sample &a )
{
	_img	= a._img;
	_row	= a._row;
	_col	= a._col;
	_width	= a._width;
	_height	= a._height;
	_weight = a._weight;

	this->originHeight = a.originHeight;
	this->originWidth = a.originWidth;
	this->scale = a.scale;
	this->sPre = a.sPre;
	this->x0 = a.x0;
	this->y0 = a.y0;
	this->xPre = a.xPre;
	this->yPre = a.yPre;
	this->XmaxBoundary = a.XmaxBoundary;
	this->YmaxBoundary = a.YmaxBoundary;


	return (*this);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void				SampleSet::resizeFtrs(int nftr)
{
	_ftrVals.resize(nftr);
	int nsamp = _samples.size();

	if( nsamp>0 )
	for(int k=0; k<nftr; k++) 
		_ftrVals[k].Resize(1,nsamp);
}

inline void				SampleSet::push_back(Matrixu *img, int x, int y, int width, int height, float weight) 
{ 
	Sample s(img,y,x,width,height, weight); 
	push_back(s); 
}


#endif