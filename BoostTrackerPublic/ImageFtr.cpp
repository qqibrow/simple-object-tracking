// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "ImageFtr.h"
#include "Sample.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
				HaarFtrParams::HaarFtrParams() 
{
	_numCh = -1;
	for( int k=0; k<1024; k++ )
		_useChannels[k] = -1;
	_minNumRect	= 2;
	_maxNumRect	= 6;
	_useChannels[0] = 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
				HaarFtr::HaarFtr()
{
	_width = 0;
	_height = 0;
	_channel = 0;
}


void			HaarFtr::generate(FtrParams *op)
{
	HaarFtrParams *p = (HaarFtrParams*)op;
	_width = p->_width;
	_height = p->_height;
	int numrects = randint(p->_minNumRect,p->_maxNumRect);
	_rects.resize(numrects);
	_weights.resize(numrects);
	_rsums.resize(numrects);
	_maxSum = 0.0f;

	for( int k=0; k<numrects; k++ )
	{
		_weights[k] = randfloat()*2-1;
		_rects[k].x = randint(0,(uint)(p->_width-3));
		_rects[k].y = randint(0,(uint)(p->_height-3));
		_rects[k].width = randint(1,(p->_width-_rects[k].x-2));
		_rects[k].height = randint(1 ,(p->_height-_rects[k].y-2));
		_rsums[k] = abs(_weights[k]*(_rects[k].width+1)*(_rects[k].height+1)*255);
		//_rects[k].width = randint(1,3);
		//_rects[k].height = randint(1,3);
	}

	if( p->_numCh < 0 ){
		p->_numCh=0;
		for( int k=0; k<1024; k++ )
			p->_numCh += p->_useChannels[k]>=0;
	}

	_channel = p->_useChannels[randint(0,p->_numCh-1)];
}

Matrixu			HaarFtr::toViz()
{
	Matrixu v(_height,_width,3);
	v.Set(0);
	v._keepIpl = true;

	for( uint k=0; k<_rects.size(); k++ )
	{
		if( _weights[k] < 0 )
			v.drawRect(_rects[k],1,(int)(255*max(-1*_weights[k],0.5)),0,0);
		else
			v.drawRect(_rects[k],1,0,(int)(255*max(_weights[k],0.5)),(int)(255*max(_weights[k],0.5)));
	}

	v._keepIpl = false;
	return v;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////
void			Ftr::compute( SampleSet &samples, const vecFtr &ftrs)
{
	int numftrs = ftrs.size();
	int numsamples = samples.size();
	if( numsamples==0 ) return;

	samples.resizeFtrs(numftrs);

	#pragma omp parallel for
	for( int ftr=0; ftr<numftrs; ftr++ ){
		//#pragma omp parallel for
		for( int k=0; k<numsamples; k++ ){
			samples.getFtrVal(k,ftr) = ftrs[ftr]->compute(samples[k]);
		}
	}

}
void			Ftr::compute( SampleSet &samples, Ftr *ftr, int ftrind )
{

	int numsamples = samples.size();

	#pragma omp parallel for
	for( int k=0; k<numsamples; k++ ){
		samples.getFtrVal(k,ftrind) = ftr->compute(samples[k]);
	}

}
vecFtr			Ftr::generate( FtrParams *params, uint num )
{
	vecFtr ftrs;

	ftrs.resize(num);
	for( uint k=0; k<num; k++ ){
		switch( params->ftrType() ){
			case 0: ftrs[k] = new HaarFtr(); break;
		}
		ftrs[k]->generate(params);
	}

	// DEBUG
	if( 0 )
		Ftr::toViz(ftrs,"ftrs");

	return ftrs;
}

void			Ftr::deleteFtrs( vecFtr ftrs )
{
	for( uint k=0; k<ftrs.size(); k ++ )
		delete ftrs[k];
}
void			Ftr::toViz( vecFtr &ftrs, const char *dirname )
{
	char fname[1024];
	Matrixu img;
	for( uint k=0; k<ftrs.size(); k++ ){
		sprintf_s(fname,"%s/ftr%05d.png",dirname,k);
		img = ftrs[k]->toViz();
		img.SaveImage(fname);
	}
}

