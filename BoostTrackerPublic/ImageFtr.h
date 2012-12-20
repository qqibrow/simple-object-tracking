// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef H_IMGFTR
#define H_IMGFTR

#include "Matrix.h"
#include "Public.h"
#include "Sample.h"

class Ftr;
typedef vector<Ftr*> vecFtr;

//////////////////////////////////////////////////////////////////////////////////////////////////////////

class FtrParams
{
public:
	uint				_width, _height;

public:
	virtual int			ftrType()=0;
};

class HaarFtrParams : public FtrParams
{
public:
						HaarFtrParams();
	uint				_maxNumRect, _minNumRect;
	int					_useChannels[1024];
	int					_numCh;

public:
	virtual int			ftrType() { return 0; };
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////
class Ftr
{
public:
	uint					_width, _height;

	virtual float			compute( const Sample &sample ) const =0;
	virtual void			generate( FtrParams *params ) = 0;
	virtual Matrixu			toViz() {Matrixu empty; return empty;};
	virtual bool			update(const SampleSet &posx, const SampleSet &negx, const Matrixf &posw, const Matrixf &negw){return false;};
	

	static void				compute( SampleSet &samples, const vecFtr &ftrs );
	static void				compute( SampleSet &samples, Ftr *ftr, int ftrind );
	static vecFtr			generate( FtrParams *params, uint num );
	static void 			deleteFtrs( vecFtr ftrs );
	static void				toViz( vecFtr &ftrs, const char *dirname );

	virtual int				ftrType()=0;
};

class HaarFtr : public Ftr
{
public:
	uint					_channel;
	vectorf					_weights;
	vector<IppiRect>		_rects;
	vectorf					_rsums;
	double					_maxSum;
	static StopWatch		_sw;

public:
							//HaarFtr( HaarFtrParams &params );
							HaarFtr();
	

	HaarFtr&				operator= ( const HaarFtr &a );
	
	float					expectedValue() const;

	virtual float			compute( const Sample &sample ) const;
	virtual void			generate( FtrParams *params );
	virtual Matrixu			toViz();
	virtual int				ftrType() { return 0; };
	
	
	
};




//////////////////////////////////////////////////////////////////////////////////////////////////////////

inline float				HaarFtr::compute( const Sample &sample ) const
{
	if( !sample._img->isInitII() ) abortError(__LINE__,__FILE__,"Integral image not initialized before called compute()");
	IppiRect r;
	float sum = 0.0f;

	//#pragma omp parallel for
	for( int k=0; k<(int)_rects.size(); k++ )
	{
		r = _rects[k];
		r.x += sample._col; r.y += sample._row;
		sum += _weights[k]*sample._img->sumRect(r,_channel);///_rsums[k];
	}

	r.x = sample._col;
	r.y = sample._row;
	r.width = (int)sample._weight;
	r.height = (int)sample._height;

	return (float)(sum);
	//return (float) (100*sum/sample._img->sumRect(r,_channel));
}


inline HaarFtr&				HaarFtr::operator= ( const HaarFtr &a )
{
	_width		= a._width;
	_height		= a._height;
	_channel	= a._channel;
	_weights	= a._weights;
	_rects		= a._rects;
	_maxSum		= a._maxSum;

	return (*this);
}

inline float				HaarFtr::expectedValue() const
{
	float sum=0.0f;
	for( int k=0; k<(int)_rects.size(); k++ ){
		sum += _weights[k]*_rects[k].height*_rects[k].width*125;
	}
	return sum;
}




#endif