// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef ONLINEBOOST_H
#define ONLINEBOOST_H

#include "Public.h"
#include "ImageFtr.h"


class ClfWeak;
class ClfStrong;
class ClfAdaBoost;
class ClfMilBoost;




class ClfStrongParams
{
public:
						ClfStrongParams(){_weakLearner = "stump"; _lRate=0.85f; _storeFtrHistory=false;};
	virtual int			clfType()=0; // [0] Online AdaBoost (Oza/Grabner) [1] Online StochBoost_LR [2] Online StochBoost_MIL
public:
	FtrParams			*_ftrParams;
	string				_weakLearner; // "stump" or "wstump"; current code only uses "stump"
	float				_lRate; // learning rate for weak learners;
	bool				_storeFtrHistory;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////

class ClfStrong
{
public:
	ClfStrongParams		*_params;
	vecFtr				_ftrs;
	vecFtr				_selectedFtrs;
	StopWatch			_clfsw;
	Matrixf				_ftrHist;
	uint				_counter;

public:
	int					nFtrs() {return _ftrs.size();};

	// abstract functions
	virtual void		init(ClfStrongParams *params)=0;
	virtual void		update(SampleSet &posx, SampleSet &negx)=0;
	virtual vectorf		classify(SampleSet &x, bool logR=true)=0;

	static ClfStrong*	makeClf(ClfStrongParams *clfparams);
	static Matrixf		applyToImage(ClfStrong *clf, Matrixu &img, bool logR=true); // returns a probability map (or log odds ratio map if logR=true)

	static void			eval(vectorf ppos, vectorf pneg, float &err, float &fp, float &fn, float thresh=0.5f);
	static float		likl(vectorf ppos, vectorf pneg);
};



//////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMEPLEMENTATIONS - PARAMS

class ClfAdaBoostParams : public ClfStrongParams
{
public:
	int					_numSel, _numFeat;

public:
						ClfAdaBoostParams(){_numSel=50;_numFeat=250;};
	virtual int			clfType() { return 0; };
};


class ClfMilBoostParams : public ClfStrongParams
{
public:
	int					_numFeat,_numSel;

	bool					ifPF;

public:
						ClfMilBoostParams(){_numSel=50;_numFeat=250;};
	virtual int			clfType() { return 1; };
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMEPLEMENTATIONS - CLF

class ClfAdaBoost : public ClfStrong
{
private:
	vectorf				_alphas;
	vectori				_selectors;
	vector<ClfWeak*>	_weakclf;
	uint				_numsamples;
	float				_sumAlph;
	vector<vectorf>		_countFPv, _countFNv, _countTPv, _countTNv; //[selector][feature]
	ClfAdaBoostParams	*_myParams;
public:
						ClfAdaBoost(){};
	virtual void		init(ClfStrongParams *params);
	virtual void		update(SampleSet &posx, SampleSet &negx);
	virtual vectorf		classify(SampleSet &x, bool logR=true);
};


class ClfMilBoost : public ClfStrong
{
private:
	vectori				_selectors;
	vector<ClfWeak*>	_weakclf;
	uint				_numsamples;
	ClfMilBoostParams	*_myParams;

public:
						ClfMilBoost(){};
	virtual void		init(ClfStrongParams *params);
	virtual void		update(SampleSet &posx, SampleSet &negx);
	virtual vectorf		classify(SampleSet &x, bool logR=true);

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// WEAK CLF

class ClfWeak
{
public:
						ClfWeak();
						ClfWeak(int id);

	virtual void		init()=0;
	virtual void		update(SampleSet &posx, SampleSet &negx, vectorf *posw=NULL, vectorf *negw=NULL)=0;
	virtual bool		classify(SampleSet &x, int i)=0;
	virtual float		classifyF(SampleSet &x, int i)=0;
	virtual void		copy(const ClfWeak* c)=0;
	
	virtual vectorb		classifySet(SampleSet &x);
	virtual vectorf		classifySetF(SampleSet &x);

	float				ftrcompute(const Sample &x) {return _ftr->compute(x);};
	float				getFtrVal(const SampleSet &x,int i) { return (x.ftrsComputed()) ? x.getFtrVal(i,_ind) : _ftr->compute(x[i]); };

protected:
	bool				_trained;
	Ftr					*_ftr;
	vecFtr				*_ftrs;
	int					_ind;
	float				_lRate;
	ClfStrong			*_parent;

friend ClfAdaBoost;
friend ClfMilBoost;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
class ClfOnlineStump : public ClfWeak
{
public:
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// members
	float				_mu0, _mu1, _sig0, _sig1;
	float				_q;
	int					_s;
	float				_n1, _n0;
	float				_e1, _e0;
public:
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// functions
						ClfOnlineStump() : ClfWeak() {init();};
						ClfOnlineStump(int ind) : ClfWeak(ind) {init();};
	virtual void		init();
	virtual void		update(SampleSet &posx, SampleSet &negx, vectorf *posw=NULL, vectorf *negw=NULL);
	virtual bool		classify(SampleSet &x, int i);
	virtual float		classifyF(SampleSet &x, int i);
	virtual void		copy(const ClfWeak* c);


};

class ClfWStump : public ClfWeak
{
public:
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// members
	float				_mu0, _mu1, _sig0, _sig1;
	float				_q;
	int					_s;
	float				_n1, _n0;
	float				_e1, _e0;
public:
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// functions
						ClfWStump() : ClfWeak() {init();};
						ClfWStump(int ind) : ClfWeak(ind) {init();};
	virtual void		init();
	virtual void		update(SampleSet &posx, SampleSet &negx, vectorf *posw=NULL, vectorf *negw=NULL);
	virtual bool		classify(SampleSet &x, int i){return classifyF(x,i)>0;};
	virtual float		classifyF(SampleSet &x, int i);
	virtual void		copy(const ClfWeak* c);
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline vectorb			ClfWeak::classifySet(SampleSet &x)
{
	vectorb res(x.size());
	
	//#pragma omp parallel for
	for( int k=0; k<(int)res.size(); k++ ){
		res[k] = classify(x,k);
	}
	return res;
}
inline vectorf			ClfWeak::classifySetF(SampleSet &x)
{
	vectorf res(x.size());
	
	#pragma omp parallel for
	for( int k=0; k<(int)res.size(); k++ ){
		res[k] = classifyF(x,k);
	}
	return res;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void				ClfOnlineStump::update(SampleSet &posx, SampleSet &negx, vectorf *posw, vectorf *negw)
{
	float posmu=0.0,negmu=0.0;
	if( posx.size()>0 ) posmu=posx.ftrVals(_ind).Mean();
	if( negx.size()>0 ) negmu=negx.ftrVals(_ind).Mean();

	if( _trained ){
		if( posx.size()>0 ){
			_mu1	= ( _lRate*_mu1  + (1-_lRate)*posmu );
			_sig1	= ( _lRate*_sig1 + (1-_lRate)* ( (posx.ftrVals(_ind)-_mu1).Sqr().Mean() ) );
		}
		if( negx.size()>0 ){
			_mu0	= ( _lRate*_mu0  + (1-_lRate)*negmu );
			_sig0	= ( _lRate*_sig0 + (1-_lRate)* ( (negx.ftrVals(_ind)-_mu0).Sqr().Mean() ) );
		}

		_q = (_mu1-_mu0)/2;
		_s = sign(_mu1-_mu0);
		_n0 = 1.0f/pow(_sig0,0.5f);
		_n1 = 1.0f/pow(_sig1,0.5f);
		_e1 = -1.0f/(2.0f*_sig1+1e-99f);
		_e0 = -1.0f/(2.0f*_sig0+1e-99f);
	}
	else{
		_trained = true;
		if( posx.size()>0 ){
			_mu1 = posmu;
			_sig1 = posx.ftrVals(_ind).Var()+1e-9f;
		}
		
		if( negx.size()>0 ){
			_mu0 = negmu;
			_sig0 = negx.ftrVals(_ind).Var()+1e-9f;
		}

		_q = (_mu1-_mu0)/2;
		_s = sign(_mu1-_mu0);
		_n0 = 1.0f/pow(_sig0,0.5f);
		_n1 = 1.0f/pow(_sig1,0.5f);
		_e1 = -1.0f/(2.0f*_sig1+1e-99f);
		_e0 = -1.0f/(2.0f*_sig0+1e-99f);
	}
}

inline bool				ClfOnlineStump::classify(SampleSet &x, int i)
{
	float xx = getFtrVal(x,i);
	double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
	double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
	bool r = p1>p0;
	return r;

	//return (_s*sign(x-_q))>0? 1 : 0 ;
}
inline float			ClfOnlineStump::classifyF(SampleSet &x, int i)
{
	float xx = getFtrVal(x,i);
	double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
	double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
	float r = (float)(log(1e-5+p1)-log(1e-5+p0));
	//r = (float)p1>p0;
	return r;

	//return (_s*sign(x-_q))>0? 1 : 0 ;
}
inline void				ClfOnlineStump::copy(const ClfWeak* c)
{
	ClfOnlineStump *cc = (ClfOnlineStump*)c;
	_mu0	= cc->_mu0;
	_mu1	= cc->_mu1;
	_sig0	= cc->_sig0;
	_sig1	= cc->_sig1;
	_lRate	= cc->_lRate;
	_e0		= cc->_e0;
	_e1		= cc->_e1;
	_n0		= cc->_n0;
	_n1		= cc->_n1;

	return;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void				ClfWStump::update(SampleSet &posx, SampleSet &negx, vectorf *posw, vectorf *negw)
{
	Matrixf poswm, negwm, poswn, negwn;
	if( (posx.size() != posw->size()) || (negx.size() != negw->size()) )
		abortError(__LINE__,__FILE__,"ClfWStump::update - number of samples and number of weights mismatch");

	float posmu=0.0, negmu=0.0;
	if( posx.size()>0 ) {
		poswm = *posw;
		poswn = poswm.normalize();
		posmu = posx.ftrVals(_ind).MeanW(poswn);
	}
	if( negx.size()>0 ) {
		negwm = *negw;
		negwn = negwm.normalize();
		negmu = negx.ftrVals(_ind).MeanW(negwn);
	}

	if( _trained ){
		if( posx.size()>0 ){
			_mu1	= ( _lRate*_mu1  + (1-_lRate)*posmu );
			_sig1	= ( _lRate*_sig1  + (1-_lRate)*posx.ftrVals(_ind).VarW(poswn,&_mu1) );
		}
		if( negx.size()>0 ){
			_mu0	= ( _lRate*_mu0  + (1-_lRate)*negmu );
			_sig0	= ( _lRate*_sig0  + (1-_lRate)*negx.ftrVals(_ind).VarW(negwn,&_mu0) );
		}
	}
	else{
		_trained = true;
		_mu1 = posmu;
		_mu0 = negmu;
		if( negx.size()>0 ) _sig0 = negx.ftrVals(_ind).VarW(negwn,&negmu)+1e-9f;
		if( posx.size()>0 ) _sig1 = posx.ftrVals(_ind).VarW(poswn,&posmu)+1e-9f;
	}

	_n0 = 1.0f/pow(_sig0,0.5f);
	_n1 = 1.0f/pow(_sig1,0.5f);
	_e1 = -1.0f/(2.0f*_sig1);
	_e0 = -1.0f/(2.0f*_sig0);
}

inline float			ClfWStump::classifyF(SampleSet &x, int i)
{
	float xx = getFtrVal(x,i);
	double p0 = exp( (xx-_mu0)*(xx-_mu0)*_e0 )*_n0;
	double p1 = exp( (xx-_mu1)*(xx-_mu1)*_e1 )*_n1;
	float r = (float)(log(1e-5+p1)-log(1e-5+p0));
	//r = (float)(r>0);
	return r;
}
inline void				ClfWStump::copy(const ClfWeak* c)
{
	ClfWStump *cc = (ClfWStump*)c;
	_mu0	= cc->_mu0;
	_mu1	= cc->_mu1;

	return;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
inline vectorf			ClfAdaBoost::classify(SampleSet &x, bool logR)
{
	int numsamples = x.size();
	vectorf res(numsamples);
	vectorb tr;
	
	// for each selector, accumate in the res vector
	//#pragma omp parallel for
	for( int sel=0; sel<(int)_selectors.size(); sel++ ){
		tr = _weakclf[_selectors[sel]]->classifySet(x);
		#pragma omp parallel for
		for( int j=0; j<numsamples; j++ ){
			res[j] += tr[j] ?  _alphas[sel] : -_alphas[sel];
		}

	}

	// return probabilities or log odds ratio
	if( !logR ){
		#pragma omp parallel for
		for( int j=0; j<(int)res.size(); j++ ){
			res[j] = sigmoid(2*res[j]);
		}
	}

	return res;
}


inline vectorf			ClfMilBoost::classify(SampleSet &x, bool logR)
{
	int numsamples = x.size();
	vectorf res(numsamples);
	vectorf tr;
	
	for( uint w=0; w<_selectors.size(); w++ ){
		tr = _weakclf[_selectors[w]]->classifySetF(x);
		#pragma omp parallel for
		for( int j=0; j<numsamples; j++ ){
			res[j] += tr[j];
		}

	}

	// return probabilities or log odds ratio
	if( !logR ){
		#pragma omp parallel for
		for( int j=0; j<(int)res.size(); j++ ){
			res[j] = sigmoid(res[j]);
		}
	}

	return res;
}





////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void				ClfStrong::eval(vectorf ppos, vectorf pneg, float &err, float &fp, float &fn, float thresh)
{
	fp=0; fn=0;
	for( uint k=0; k<ppos.size(); k++ )
		(ppos[k] < thresh) ? fn++ : fn;

	for( uint k=0; k<pneg.size(); k++ )
		(pneg[k] >= thresh) ? fp++ : fp;

	fn /= ppos.size();
	fp /= pneg.size();

	err = 0.5f*fp + 0.5f*fn;
}
inline float			ClfStrong::likl(vectorf ppos, vectorf pneg)
{
	float likl=0, posw = 1.0f/ppos.size(), negw = 1.0f/pneg.size();

	for( uint k=0; k<ppos.size(); k++ )
		likl += log(ppos[k]+1e-5f)*posw;

	for( uint k=0; k<pneg.size(); k++ )
		likl += log(1-pneg[k]+1e-5f)*negw;

	return likl;
}





#endif