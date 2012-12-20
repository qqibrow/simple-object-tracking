// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Sample.h"

bool Sample::operator < (const Sample& s1 ) const 
{ 
	bool flag = this->_weight > s1._weight;
	return flag;
};

void Sample::transition(Matrixu *img)
{
	  float tempX, tempY, tempS;
	  
      /* calculate the x, y, s for the next step */
      /* sample new state using second-order autoregressive dynamics */
       
	  this->_img = img;
	  do{
			//tempX = A1 * ( _col - x0 ) + A2 * ( xPre - x0 ) +
			//B0 * randgaus( 0, TRANS_X_STD ) + x0;	

			///* make sure that pn.x >= 0 && pn.x <= w -1 */
		 // tempX = max( 0.0, min( XmaxBoundary, tempX ) );

		 // tempY = A1 * ( _row - y0 ) + A2 * ( yPre - y0 ) +
			//B0 * randgaus( 0, TRANS_Y_STD ) + y0;
		 // tempY = max( 0.0, min( YmaxBoundary, tempY ) );

		 // tempS = A1 * ( scale - 1.0 ) + A2 * ( sPre - 1.0 ) +
			//B0 * randgaus( 0, TRANS_S_STD ) + 1.0;
		 // tempS = max( 0.1, tempS );

		    tempX = cvRound((float)this->_col + randgaus( 0, TRANS_X_STD ));
            tempY =  cvRound((float)this->_row + randgaus( 0, TRANS_Y_STD ));
			tempS =  B0 * randgaus( 0, TRANS_S_STD ) + 1.0;

	  }while(!(tempX > 0 && tempX < XmaxBoundary && tempY > 0 && tempY < YmaxBoundary && tempS > 0.3 && tempS <=1.0 ));

	 /* update data */

	  this->xPre = this->_col;
	  this->yPre = this->_row;
	  this->sPre = this->scale;

	  this->_col = tempX;
	  this->_row = tempY;
	  this->scale = tempS;

	  this->_width = cvRound( this->originWidth * scale);
	  this->_height  = cvRound(this->originHeight *scale );


	//this->_img = img;
	//int tempX, tempY;

	//do{

 //   tempX = cvRound((float)this->_col + randgaus( 0, TRANS_X_STD ));
 //   tempY =  cvRound((float)this->_row + randgaus( 0, TRANS_Y_STD ));

	//}while( !(tempX > 0 && tempX < XmaxBoundary && tempY > 0 && tempY < YmaxBoundary) );
 // 
	//    this->_col =  tempX;
 //      this->_row  =  tempY;

      

	  //权值没变唉   width, height 一直没赋值，我现在还不知道这两个是甘什么用的。
}



//generate particles
// not let the particles to overcome the boundaries

void SampleSet::init_particle_distributions(Matrixu *img, int x, int y, int w, int h, int maxnum)
{

	/* static set ImageSize 应取的值
	int rowsz = img->rows() - h - 1;
	int colsz = img->cols() - w - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;

	uint minrow = max(0,(int)y-(int)inrad);
    uint maxrow = min((int)rowsz-1,(int)y+(int)inrad);
    uint mincol = max(0,(int)x-(int)inrad);
    uint maxcol = min((int)colsz-1,(int)x+(int)inrad);

	//fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

	_samples.resize( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	float prob = ((float)(maxnum))/_samples.size();
	int 
   */

	_samples.resize(maxnum);
	for(int k = 0; k < maxnum; k++)
	{
		//generate particle
	  _samples[k]._img = img;
	  _samples[k]._col =  x;
	  _samples[k]._row = y;
	   _samples[k].x0 = x;
       _samples[k].xPre = x;
	   _samples[k].y0  = y;
	  _samples[k].yPre = y ;

	  _samples[k].sPre = _samples[k].scale = 1.0;
	  _samples[k]._width = _samples[k].originWidth = w;
	  _samples[k]._height = _samples[k].originHeight= h;
	  _samples[k].XmaxBoundary = img->cols() - w - 1;
	  _samples[k].YmaxBoundary = img->rows() - h - 1;
	  _samples[k]._weight = 1.0;

	}
}

void SampleSet::sampleParticles(Matrixu *img)
{
	for (int i = 0; i < _samples.size(); i++)
	{
		_samples[i].transition(img);
	}
}




void SampleSet::resample_1()
{
	int size = _samples.size();
   static float newWeight = 1.0;

	/* 
	init newSamples, the new Samples vector will save the pointers of the samples in _samples. different pointers may 
	point to the same sample in _samples. the is the main point of resample: let the important sample to multiply
	*/
	
	newSamples.resize(size);
    vectorf CDF(size);

	/* firstly, calculate the CDF */
	CDF[0] = _samples[0]._weight;

	for( int i = 1; i < size; i++)
	{
		CDF[i] = CDF[i-1] + _samples[i]._weight;
	}


	
	int newSamplesIndex = 0;
	float noise;
	for(int i = 0 ; i < size; i++)
	{
        newSamplesIndex = 0;
		noise = randfloat(); 

		/* then find the first sample whose related CDF weights noise and give it to newSamples[i]*/
		while( noise > CDF[newSamplesIndex])
			newSamplesIndex++;
        
		
		if( newSamplesIndex < size )
		{
			newSamples[i] = _samples[newSamplesIndex];
		}
	}

	sort( newSamples.begin(), newSamples.end());
	for( int i = 0 ; i < size; i++)
	{
		newSamples[i]._weight = newWeight;
	}
	_samples = newSamples;
}

void SampleSet::resample_2()
{
	int size = _samples.size();
	newSamples.resize(size);
	
    int numOfOneParticle;
    int newSamplesIndex = 0;

	static float newWeight = 1.0;

    // firstly, sort the _samples according to weights  
	sort( _samples.begin(), _samples.end());
    

	// secondly, for every particle, multiply it according to its weight 
    for(int oldSampleIndex = 0; oldSampleIndex < size; oldSampleIndex++ )
    {
		numOfOneParticle = cvRound( _samples[ oldSampleIndex ]._weight * size );
        for(int j = 0; j < numOfOneParticle; j++ )
		{
		  newSamples[ newSamplesIndex ] = _samples[ oldSampleIndex ];

		  // weight turns to be 1/n
		  newSamples[newSamplesIndex]._weight = newWeight;
		  newSamplesIndex++;

		  if( newSamplesIndex == size )
			goto exit;
		}
    }

  // if newSamples has not reached size, then multiply  _samples[ 0 ], whose weight largest.
    while( newSamplesIndex < size )
	{
      newSamples[ newSamplesIndex ] = _samples[ 0 ];
	  newSamples[newSamplesIndex]._weight = newWeight;
	  newSamplesIndex++;

	}

 exit:
	_samples = newSamples;
  return ;
}


void SampleSet::weightParticles( vectorf weights)
{
	assert( size() == weights.size());
	for( int i = 0 ; i < size(); i++ )
	{
		_samples[i]._weight = weights[i];
	}
}



//注意sum有可能为负
void SampleSet::normalize_weights()
{
	  int n = size();
	  float sum = 0;
	  int i = 0;

	  for( i = 0; i < n; i++ )
		sum += _samples[i]._weight;
	  for( i = 0; i < n; i++ )
		 _samples[i]._weight /= sum;
}

Sample SampleSet::findBestParticle()
{
	   // firstly, sort the _samples according to weights  
	//sort( _samples.begin(), _samples.end());
	return _samples[0];

}

Sample SampleSet::findMeanParticle()
{
	static const int mean = 5;
	Sample meanParticle;
	meanParticle._col = 0;
	meanParticle._row = 0;
	meanParticle._width = _samples[0]._width;
	meanParticle._height = _samples[0]._height;

	for( int i = 0; i < mean; i++)
	{
		meanParticle._col += _samples[i]._col;
		meanParticle._row += _samples[i]._row;
	}

	meanParticle._col = meanParticle._col/5;
	meanParticle._row = meanParticle._row/5;

	return meanParticle;

}
				Sample::Sample(Matrixu *img, int row, int col, int width, int height, float weight) 
{
	_img	= img;
	_row	= row;
	_col	= col;
	_width	= width;
	_height	= height;
	_weight = weight;
}



void		SampleSet::sampleImage(Matrixu *img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
{
	int rowsz = img->rows() - h - 1;
	int colsz = img->cols() - w - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;

	uint minrow = max(0,(int)y-(int)inrad);
    uint maxrow = min((int)rowsz-1,(int)y+(int)inrad);
    uint mincol = max(0,(int)x-(int)inrad);
    uint maxcol = min((int)colsz-1,(int)x+(int)inrad);

	//fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

	_samples.resize( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	float prob = ((float)(maxnum))/_samples.size();

	//#pragma omp parallel for
	for( int r=minrow; r<=(int)maxrow; r++ )
		for( int c=mincol; c<=(int)maxcol; c++ ){
			dist = (y-r)*(y-r) + (x-c)*(x-c);
			if( randfloat()<prob && dist < inradsq && dist >= outradsq ){
				_samples[i]._img = img;
				_samples[i]._col = c;
				_samples[i]._row = r;
				_samples[i]._height = h;
				_samples[i]._width = w;
				i++;
			}
		}

	_samples.resize(min(i,maxnum));

}

void		SampleSet::sampleImage(Matrixu *img, uint num, int w, int h)
{
	int rowsz = img->rows() - h - 1;
	int colsz = img->cols() - w - 1;

	_samples.resize( num );
	//#pragma omp parallel for
	for( int i=0; i<(int)num; i++ ){
		_samples[i]._img = img;
		_samples[i]._col = randint(0,colsz);
		_samples[i]._row = randint(0,rowsz);
		_samples[i]._height = h;
		_samples[i]._width = w;
	}
}