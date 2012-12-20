// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Sample.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


ClfStrong*			ClfStrong::makeClf(ClfStrongParams *clfparams)
{
	ClfStrong* clf;

	switch(clfparams->clfType()){
		case 0:
			clf = new ClfAdaBoost();
			break;
		case 1:
			clf = new ClfMilBoost();
			break;

		default:
			abortError(__LINE__,__FILE__,"Incorrect clf type!");
	}

	clf->init(clfparams);
	return clf;
}

Matrixf				ClfStrong::applyToImage(ClfStrong *clf, Matrixu &img, bool logR)
{
	img.initII();
	Matrixf resp(img.rows(),img.cols());
	int height = clf->_params->_ftrParams->_height;
	int width = clf->_params->_ftrParams->_width;

	int rowsz = img.rows() - width - 1;
	int colsz = img.cols() - height - 1;

	SampleSet x; x.sampleImage(&img,0,0,width,height,100000); // sample every point
	Ftr::compute(x,clf->_ftrs);
	vectorf rf = clf->classify(x,logR);
	for( int i=0; i<x.size(); i++ )
		resp(x[i]._row,x[i]._col) = rf[i];

	//#pragma omp parallel for
	//for( int r=0; r<rowsz; r++ ){
	//	#pragma omp parallel for
	//	for( int c=0; c<colsz; c++ ){
	//			SampleSet x; vectorf rf;
	//			x.push_back(&img,r,c,width,height);
	//			rf = clf->classify(x);
	//			resp(r,c) = rf[0];
	//			x.clear();
	//	}
	//}

	return resp;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
					ClfWeak::ClfWeak()
{
	_trained=false; _ind=-1;
}

					ClfWeak::ClfWeak(int id)
{
	_trained=false; _ind=id;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void				ClfOnlineStump::init()
{
	_mu0	= 0;
	_mu1	= 0;
	_sig0	= 1;
	_sig1	= 1;
	_lRate	= 0.85f;
	_trained = false;
}

void				ClfWStump::init()
{
	_mu0	= 0;
	_mu1	= 0;
	_sig0	= 1;
	_sig1	= 1;
	_lRate	= 0.85f;
	_trained = false;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
void				ClfAdaBoost::init(ClfStrongParams *params)
{
	// initialize model
	_params		= params;
	_myParams	= (ClfAdaBoostParams*)params;
	_numsamples = 0;

	if( _myParams->_numSel > _myParams->_numFeat  || _myParams->_numSel < 1 )
		_myParams->_numSel = _myParams->_numFeat/2;

	//_countFP.Resize(_params._numSel,_params._numFeat,1);
	//_countFN.Resize(_params._numSel,_params._numFeat,1);
	//_countTP.Resize(_params._numSel,_params._numFeat,1);
	//_countTN.Resize(_params._numSel,_params._numFeat,1);
	resizeVec(_countFPv,_myParams->_numSel, _myParams->_numFeat,1.0f);
	resizeVec(_countTPv,_myParams->_numSel, _myParams->_numFeat,1.0f);
	resizeVec(_countFNv,_myParams->_numSel, _myParams->_numFeat,1.0f);
	resizeVec(_countTNv,_myParams->_numSel, _myParams->_numFeat,1.0f);

	_alphas.resize(_myParams->_numSel,0);
	_ftrs = Ftr::generate(_myParams->_ftrParams,_myParams->_numFeat);
	_selectors.resize(_myParams->_numSel,0);
	_weakclf.resize(_myParams->_numFeat);
	for( int k=0; k<_myParams->_numFeat; k++ )
		//if (_params._weakLearner == string("kalman"))
		//	_weakclf[k] = new ClfKalmanStump();
		//else 
		if(_myParams->_weakLearner == string("stump")){
			_weakclf[k] = new ClfOnlineStump(k);
			_weakclf[k]->_ftr = _ftrs[k];
			_weakclf[k]->_lRate = _myParams->_lRate;
			_weakclf[k]->_parent = this;
		}
		else if( _myParams->_weakLearner == string("wstump")){
			_weakclf[k] = new ClfWStump(k);
			_weakclf[k]->_ftr = _ftrs[k];
			_weakclf[k]->_lRate = _myParams->_lRate;
			_weakclf[k]->_parent = this;
		}
		else
			abortError(__LINE__,__FILE__,"incorrect weak clf name");
}
void				ClfAdaBoost::update(SampleSet &posx, SampleSet &negx)
{
	_clfsw.Start();
	int numpts = posx.size() + negx.size();

	// compute ftrs
	if( !posx.ftrsComputed() ) Ftr::compute(posx, _ftrs);
	if( !negx.ftrsComputed() ) Ftr::compute(negx, _ftrs);

	//vectorf poslam(posx[0].size(),.5f*numpts/posx[0].size()), neglam(negx[0].size(),.5f*numpts/negx[0].size());
	//vectorf poslam(posx[0].size(),1), neglam(negx[0].size(),1);
	vectorf poslam(posx.size(),.5f/posx.size()), neglam(negx.size(),.5f/negx.size());
	vector<vectorb> pospred(nFtrs()), negpred(nFtrs());
	vectorf errs(nFtrs());
	vectori order(nFtrs());

	_sumAlph=0.0f;
	_selectors.clear();

	// update all weak classifiers and get predicted labels
	#pragma omp parallel for
	for( int k=0; k<nFtrs(); k++ ){
		_weakclf[k]->update(posx,negx);
		pospred[k] = _weakclf[k]->classifySet(posx);
		negpred[k] = _weakclf[k]->classifySet(negx);
	}

	vectori worstinds;

	// loop over selectors
	for( int t=0; t<_myParams->_numSel; t++ ){
		// calculate errors for selector t
		//#pragma omp parallel for
		//for( int j=0; j<(int)poslam.size(); j++ ){
		//	float curw = poslam[j];
		//	if( curw < 1e-5 ) continue;
		//	for( int k=0; k<_params._numFeat; k++ ){
		//		(pospred[k][j])? _countTPv[t][k] += curw : _countFNv[t][k] += curw;
		//	}
		//}
		//#pragma omp parallel for
		//for( int j=0; j<(int)neglam.size(); j++ ){
		//	float curw = neglam[j];
		//	if( curw < 1e-5 ) continue;
		//	for( int k=0; k<_params._numFeat; k++ ){
		//		(!negpred[k][j])? _countTNv[t][k] += curw : _countFPv[t][k] += curw;
		//	}
		//}

		#pragma omp parallel for
		for( int k=0; k<_myParams->_numFeat; k++ ){
			for( int j=0; j<(int)poslam.size(); j++ ){
				//if( poslam[j] > 1e-5 )
				(pospred[k][j])? _countTPv[t][k] += poslam[j] : _countFNv[t][k] += poslam[j];
			}
		}
		#pragma omp parallel for
		for( int k=0; k<_myParams->_numFeat; k++ ){
			for( int j=0; j<(int)neglam.size(); j++ ){
				//if( neglam[j] > 1e-5 )
				(!negpred[k][j])? _countTNv[t][k] += neglam[j] : _countFPv[t][k] += neglam[j];
			}
		}
		#pragma omp parallel for
		for( int k=0; k<_myParams->_numFeat; k++ ){
			//float fp,fn;
			//fp = _countFPv[t][k] / (_countFPv[t][k] + _countTNv[t][k]);
			//fn = _countFNv[t][k] / (_countFNv[t][k] + _countTPv[t][k]);
			//errs[k] = 0.3f*fp + 0.7f*fn;
			errs[k] = (_countFPv[t][k]+_countFNv[t][k])/(_countFPv[t][k]+_countFNv[t][k]+_countTPv[t][k]+_countTNv[t][k]);
		}

		// pick the best weak clf and udpate _selectors and _selectedFtrs
		float minerr;
		uint bestind;

		sort_order_des(errs,order);

		// find best in that isn't already included
		for( uint k=0; k<order.size(); k++ )
			if( count( _selectors.begin(), _selectors.end(), order[k])==0 )	{
				_selectors.push_back(order[k]);
				minerr = errs[k];
				bestind = order[k];
				break;
			}

		//cout << "min err=" << minerr << endl;

		// find worst ind
		worstinds.push_back(order[order.size()-1]);

		// update alpha
		_alphas[t] = max(0,min(0.5f*log((1-minerr)/(minerr+0.00001f)),10));
		_sumAlph += _alphas[t];

		// update weights
		float corw = 1/(2-2*minerr);
		float incorw = 1/(2*minerr);
		#pragma omp parallel for
		for( int j=0; j<(int)poslam.size(); j++ )
			poslam[j] *= (pospred[bestind][j]==1)? corw : incorw;
		#pragma omp parallel for
		for( int j=0; j<(int)neglam.size(); j++ )
			neglam[j] *= (negpred[bestind][j]==0)? corw : incorw;
				 
	}

	_numsamples += numpts;
	_clfsw.Stop();



	return;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
void				ClfMilBoost::init(ClfStrongParams *params)
{
	// initialize model
	_params		= params;
	_myParams	= (ClfMilBoostParams*)params;
	_numsamples = 0;


	//生成了_numFeat 个 weakfeature
	_ftrs = Ftr::generate(_myParams->_ftrParams,_myParams->_numFeat);
	if( params->_storeFtrHistory ) Ftr::toViz( _ftrs, "haarftrs" );
	_weakclf.resize(_myParams->_numFeat);
	for( int k=0; k<_myParams->_numFeat; k++ )
		if(_myParams->_weakLearner == string("stump")){
			_weakclf[k] = new ClfOnlineStump(k);
			_weakclf[k]->_ftr = _ftrs[k];
			_weakclf[k]->_lRate = _myParams->_lRate;
			_weakclf[k]->_parent = this;
		}
		else if(_myParams->_weakLearner == string("wstump")){
			_weakclf[k] = new ClfWStump(k);
			_weakclf[k]->_ftr = _ftrs[k];
			_weakclf[k]->_lRate = _myParams->_lRate;
			_weakclf[k]->_parent = this;
		}
		else
			abortError(__LINE__,__FILE__,"incorrect weak clf name");

		if( params->_storeFtrHistory )
			this->_ftrHist.Resize(_myParams->_numFeat,2000);

		_counter=0;
}
void				ClfMilBoost::update(SampleSet &posx, SampleSet &negx)
{
	_clfsw.Start();
	int numneg = negx.size();
	int numpos = posx.size();

	// compute ftrs
	if( !posx.ftrsComputed() ) Ftr::compute(posx, _ftrs);
	if( !negx.ftrsComputed() ) Ftr::compute(negx, _ftrs);

	// initialize H
	static vectorf Hpos, Hneg;
	Hpos.clear(); Hneg.clear();
	Hpos.resize(posx.size(),0.0f), Hneg.resize(negx.size(),0.0f);

	_selectors.clear();
	vectorf posw(posx.size()), negw(negx.size());
	vector<vectorf> pospred(_weakclf.size()), negpred(_weakclf.size());

	// train all weak classifiers without weights
	#pragma omp parallel for
	for( int m=0; m<_myParams->_numFeat; m++ ){
		_weakclf[m]->update(posx,negx);
		pospred[m] = _weakclf[m]->classifySetF(posx);
		negpred[m] = _weakclf[m]->classifySetF(negx);
	}

	// pick the best features
	for( int s=0; s<_myParams->_numSel; s++ ){

		// compute errors/likl for all weak clfs
		vectorf poslikl(_weakclf.size(),1.0f), neglikl(_weakclf.size()), likl(_weakclf.size());
		#pragma omp parallel for
		for( int w=0; w<(int)_weakclf.size(); w++) {
			float lll=1.0f;
			//#pragma omp parallel for reduction(*: lll)
			for( int j=0; j<numpos; j++ )
				lll *= ( 1-sigmoid(Hpos[j]+pospred[w][j]) );
			poslikl[w] = (float)-log(1-lll+1e-5);
			
			lll=0.0f;
			//#pragma omp parallel for reduction(+: lll)
			for( int j=0; j<numneg; j++ )
				lll += (float)-log(1e-5f+1-sigmoid(Hneg[j]+negpred[w][j]));
			neglikl[w]=lll;

			likl[w] = poslikl[w]/numpos + neglikl[w]/numneg;
		}

		// pick best weak clf
		vectori order;
		sort_order_des(likl,order);

		// find best weakclf that isn't already included
		for( uint k=0; k<order.size(); k++ )
			if( count( _selectors.begin(), _selectors.end(), order[k])==0 ){
				_selectors.push_back(order[k]);
				break;
			}
		
		// update H = H + h_m
		#pragma omp parallel for
		for( int k=0; k<posx.size(); k++ )
			Hpos[k] += pospred[_selectors[s]][k];
		#pragma omp parallel for
		for( int k=0; k<negx.size(); k++ )
			Hneg[k] += negpred[_selectors[s]][k];

	}

	if( _myParams->_storeFtrHistory )
		for( uint j=0; j<_selectors.size(); j++ )
			_ftrHist(_selectors[j],_counter) = 1.0f/(j+1);

	_counter++;
	_clfsw.Stop();

	return;
}













