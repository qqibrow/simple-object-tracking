// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "OnlineBoost.h"
#include "Tracker.h"
#include "Public.h"
#include "Sample.h"





CvHaarClassifierCascade* Tracker::facecascade = NULL;

bool			SimpleTracker::init(Matrixu frame, SimpleTrackerParams p, ClfStrongParams *clfparams)
{
	static Matrixu *img;

	img = &frame;
	frame.initII();

	_clf = ClfStrong::makeClf(clfparams);

	////////////////////////////////init a Adaboost //////////////////////////

	//_clf


	_curState.resize(4);
	for(int i=0;i<4;i++ ) _curState[i] = p._initstate[i];
	SampleSet posx, negx;

	fprintf(stderr,"Initializing Tracker..\n");

	// sample positives and negatives from first frame
	posx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], p._init_postrainrad);
	negx.sampleImage(img, (uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3], 2.0f*p._srchwinsz, (1.5f*p._init_postrainrad), p._init_negnumtrain);
	if( posx.size()<1 || negx.size()<1 ) return false;
     


	// init particle filter
	if(	((ClfMilBoostParams*)clfparams)->ifPF == true)
	particles.init_particle_distributions(img,(uint)_curState[0],(uint)_curState[1], (uint)_curState[2], (uint)_curState[3],PARTICLENUM);
 
	// train
	_clf->update(posx,negx);
	negx.clear();

	img->FreeII();

	_trparams = p;
	_clfparams = clfparams;
	_cnt = 0;


	return true;
}
void			SimpleTracker::track_frames(vector<Matrixu> &video, SimpleTrackerParams p, ClfStrongParams *clfparams)
{
	CvVideoWriter* w = NULL;
	Matrixu states(video.size(), 4);
	Matrixu t; string pnd="#";

	// save video file
	if( ! p._vidsave.empty() ){

	//	w = cvCreateVideoWriter( p._vidsave.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(video[0].cols(), video[0].rows()), 3 );
		w = cvCreateVideoWriter( p._vidsave.c_str(), -1, 15, cvSize(video[0].cols(), video[0].rows()), 1 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	// initialization
	int frameind = 0;
	if( p._initWithFace ){  // init with face
		fprintf(stderr,"Searching for face...\n");
		while( !Tracker::initFace(&p,video[frameind]) ){
			video[frameind].conv2RGB(t); t._keepIpl=true;
			t.drawText(("#"+int2str(frameind,3)).c_str(),1,25,255,255,0);
			// display on screen & write to disk
			if( p._disp ){ t.display(1); cvWaitKey(1); }
			if( w != NULL ) cvWriteFrame( w, t.getIpl() );
			t._keepIpl=false; t.freeIpl();
			//frameind++;
		}
		clfparams->_ftrParams->_width	= (uint)p._initstate[2];
		clfparams->_ftrParams->_height	= (uint)p._initstate[3];
		init(video[frameind], p, clfparams);
		
	} // init with params
	else{
		clfparams->_ftrParams->_width	= (uint)p._initstate[2];
		clfparams->_ftrParams->_height	= (uint)p._initstate[3];


		// initialize; training the weak classifier
		init(video[0], p, clfparams);



		states(frameind,0) = (uint)_curState[0];
		states(frameind,1) = (uint)_curState[1];
		states(frameind,2) = (uint)_curState[2];
		states(frameind,3) = (uint)_curState[3];
		//frameind++;
	}

	// track rest of frames
	StopWatch sw(true); double ttt;
	for( frameind; frameind<(int)video.size(); frameind++ )
	{
		ttt = sw.Elapsed(true);
		fprintf(stderr,"%s%d Frames/%f sec = %f FPS",ERASELINE,frameind,ttt,((double)frameind)/ttt);

		if(	((ClfMilBoostParams*)clfparams)->ifPF == true)
			PF_track_frame(video[frameind],t);
		else
			track_frame(video[frameind],t);



		if( p._disp || w!= NULL ){
			video[frameind] = t;
			video[frameind].drawText(("#"+int2str(frameind,3)).c_str(),1,25,255,255,0);
			video[frameind].createIpl();
			video[frameind]._keepIpl = true;
			// display on screen
			if( p._disp ){
				video[frameind].display(1);
				cvWaitKey(1);
			}
			// save video
			if( w != NULL && frameind<(int)video.size()-1 )
				Matrixu::WriteFrame(w, video[frameind]);
			video[frameind]._keepIpl = false;
			video[frameind].freeIpl();
		}
		
		for( int s=0; s<4; s++ ) states(frameind,s) = (uint)_curState[s];

	}

	// save states
	if( !p._trsave.empty() ){
		bool scs = states.DLMWrite(p._trsave.c_str());
		if( !scs ) abortError(__LINE__,__FILE__,"error saving states to file");
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );

}


double			SimpleTracker::PF_track_frame(Matrixu &frame, Matrixu &framedisp)
{
	static SampleSet posx, negx, detectx;
	static vectorf prob;
	static vectori order;
	static Matrixu *img;
	double resp;
	static Sample bestParticle;
	
    static int resultsIndex = 0;
	static int results[1000][4];
	memset(results,0,sizeof(int));

	// copy a color version into framedisp (this is where we will draw a colored box around the object for output)
	frame.conv2RGB(framedisp);

	img = &frame;
	frame.initII();

	// run current clf on search window
	//detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], (float)_trparams._srchwinsz);

	particles.sampleParticles(img);
	prob = _clf->classify( particles ,_trparams._useLogR);
	
	int index =  max_idx(prob);
	resp=prob[index];
	Sample proposed = particles[index];
	
    //normalize prob
	float sum = 0;
	int size = prob.size();


	for( int i = 0; i < size; i ++)
	{
		sum += prob[i];
	}
	for( int i = 0; i < size; i ++)
	  prob[i] /= sum;

	// sum <0 的时候，原来对应正的权值会变成负的，变成取不到的点，所以要*-1.
	if( sum < 0 )
	{
		for( int i = 0; i < size; i ++)
	    prob[i] = -prob[i];
	}
    int a =  max_idx(prob);
	//都加上权值最小的，原意是当让权值小于0的点分裂时，-0.5的次数会比 -1的次数多。但这样原本权值大的数会分裂更多。
	//也许让权值都 < 0 的分类一次会更好
	//float minProb = prob[ min_idx(prob)];
	//for( int i = 0; i < size; i ++)
	//    prob[i] += minProb;

	particles.weightParticles( prob );
    
//	particles.normalize_weights();
	particles.resample_2();

	// find best location
	
	
	
	//bestParticle = particles.findBestParticle();
  //  bestParticle = particles.findBestParticle();
    
	bestParticle = particles.findBestParticle();


	_curState[1] = bestParticle._row;
	_curState[0] = bestParticle._col;

	results[resultsIndex][0] = _curState[0];
	results[resultsIndex][1] = _curState[1];


	resultsIndex++;
	// train location clf (negx are randomly selected from image, posx is just the current tracker location)

	if(resultsIndex % UPDATEFREQUENCE == 0)
	{

		if( _trparams._negsamplestrat == 0 )
			negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
		else
			negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], 
				(1.5f*_trparams._srchwinsz), _trparams._posradtrain+5, _trparams._negnumtrain);

		if( _trparams._posradtrain == 1 )
			posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
		else
			posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

		_clf->update(posx,negx);
	}

	/////// DEBUG /////// display sampled negative points
	if( _trparams._debugv ){
		for( int j=0; j<negx.size(); j++ )
			framedisp.drawEllipse(1,1,(float)negx[j]._col,(float)negx[j]._row,1,255,0,255);
	}

	// clean up
	img->FreeII();
	posx.clear(); negx.clear(); detectx.clear();

//	 draw a colored box around object
	framedisp.drawRect(_curState[2], _curState[3], _curState[0], _curState[1], 1, 0,
			_trparams._lineWidth, _trparams._boxcolor[0], _trparams._boxcolor[1], _trparams._boxcolor[2] );

	_cnt++;

	return resp;
}





double			SimpleTracker::track_frame(Matrixu &frame, Matrixu &framedisp)
{
	static SampleSet posx, negx, detectx;
	static vectorf prob;
	static vectori order;
	static Matrixu *img;

	double resp;

	// copy a color version into framedisp (this is where we will draw a colored box around the object for output)
	frame.conv2RGB(framedisp);

	img = &frame;
	frame.initII();

	// run current clf on search window
	detectx.sampleImage(img,(uint)_curState[0],(uint)_curState[1],(uint)_curState[2],(uint)_curState[3], (float)_trparams._srchwinsz);
	prob = _clf->classify(detectx,_trparams._useLogR);

	/////// DEBUG /////// display actual probability map
	if( _trparams._debugv ){
		Matrixf probimg(frame.rows(),frame.cols());
		for( uint k=0; k<(uint)detectx.size(); k++ )
			probimg(detectx[k]._row, detectx[k]._col) = prob[k];

		probimg.convert2img().display(2,2);
		cvWaitKey(1);
	}

	// find best location
	int bestind = max_idx(prob);
	resp=prob[bestind];

	_curState[1] = (float)detectx[bestind]._row; 
	_curState[0] = (float)detectx[bestind]._col;

	// train location clf (negx are randomly selected from image, posx is just the current tracker location)

	if( _trparams._negsamplestrat == 0 )
		negx.sampleImage(img, _trparams._negnumtrain, (int)_curState[2], (int)_curState[3]);
	else
		negx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], 
			(1.5f*_trparams._srchwinsz), _trparams._posradtrain+5, _trparams._negnumtrain);

	if( _trparams._posradtrain == 1 )
		posx.push_back(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3]);
	else
		posx.sampleImage(img, (int)_curState[0], (int)_curState[1], (int)_curState[2], (int)_curState[3], _trparams._posradtrain, 0, _trparams._posmaxtrain);

	_clf->update(posx,negx);

	/////// DEBUG /////// display sampled negative points
	if( _trparams._debugv ){
		for( int j=0; j<negx.size(); j++ )
			framedisp.drawEllipse(1,1,(float)negx[j]._col,(float)negx[j]._row,1,255,0,255);
	}

	// clean up
	img->FreeII();
	posx.clear(); negx.clear(); detectx.clear();

	// draw a colored box around object
	framedisp.drawRect(_curState[2], _curState[3], _curState[0], _curState[1], 1, 0,
			_trparams._lineWidth, _trparams._boxcolor[0], _trparams._boxcolor[1], _trparams._boxcolor[2] );

	_cnt++;

	return resp;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void			Tracker::replayTracker(vector<Matrixu> &vid, string statesfile, string outputvid, uint R, uint G, uint B)
{
	Matrixf states;
	states.DLMRead(statesfile.c_str());
	Matrixu colorframe;

	// save video file
	CvVideoWriter* w = NULL;
	if( ! outputvid.empty() ){
		w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	for( uint k=0; k<vid.size(); k++ )
	{	
		vid[k].conv2RGB(colorframe);
		colorframe.drawRect(states(k,2),states(k,3),states(k,0),states(k,1),1,0,2,R,G,B);
		colorframe.drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
		colorframe._keepIpl=true;
		colorframe.display(1,2);
		cvWaitKey(1);
		if( w != NULL )
			cvWriteFrame( w, colorframe.getIpl() );
		colorframe._keepIpl=false; colorframe.freeIpl();
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
}
void			Tracker::replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors)
{
	Matrixu states;
	vector<Matrixu> resvid(vid.size());
	Matrixu colorframe;

	// save video file
	CvVideoWriter* w = NULL;
	if( ! outputvid.empty() ){
		w = cvCreateVideoWriter( outputvid.c_str(), CV_FOURCC('I','Y','U','V'), 15, cvSize(vid[0].cols(), vid[0].rows()), 3 );
		if( w==NULL ) abortError(__LINE__,__FILE__,"Error opening video file for output");
	}

	for( uint k=0; k<vid.size(); k++ ){
		vid[k].conv2RGB(resvid[k]);
		resvid[k].drawText(("#"+int2str(k,3)).c_str(),1,25,255,255,0);
	}

	for( uint j=0; j<statesfile.size(); j++ ){
		states.DLMRead(statesfile[j].c_str());
		for( uint k=0; k<vid.size(); k++ )	
			resvid[k].drawRect(states(k,3),states(k,2),states(k,0),states(k,1),1,0,3,colors(j,0),colors(j,1),colors(j,2));
	}

	for( uint k=0; k<vid.size(); k++ ){
		resvid[k]._keepIpl=true;
		resvid[k].display(1,2);
		cvWaitKey(1);
		if( w!=NULL && k<vid.size()-1)
			Matrixu::WriteFrame(w, resvid[k]);
		resvid[k]._keepIpl=false; resvid[k].freeIpl();
	}

	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
}
bool			Tracker::initFace(TrackerParams* params, Matrixu &frame)
{
	const char* cascade_name = "haarcascade_frontalface_alt_tree.xml";
	const int minsz = 20;
	if( Tracker::facecascade == NULL )
		Tracker::facecascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

	frame.createIpl();
	IplImage *img = frame.getIpl();
	IplImage* gray = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1 );
    cvCvtColor(img, gray, CV_BGR2GRAY );
	frame.freeIpl();
	cvEqualizeHist(gray, gray);

	CvMemStorage* storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);
	CvSeq* faces = cvHaarDetectObjects(gray, Tracker::facecascade, storage, 1.05, 3, CV_HAAR_DO_CANNY_PRUNING ,cvSize(minsz, minsz));
	
	int index = faces->total-1;
	CvRect* r = (CvRect*)cvGetSeqElem( faces, index );
	
	

	while(r && (r->width<minsz || r->height<minsz || (r->y+r->height+10)>frame.rows() || (r->x+r->width)>frame.cols() ||
		r->y<0 || r->x<0)){
		r = (CvRect*)cvGetSeqElem( faces, --index);
	}

	//if( r == NULL ){
	//	cout << "ERROR: no face" << endl;
	//	return false;
	//}
	//else 
	//	cout << "Face Found: " << r->x << " " << r->y << " " << r->width << " " << r->height << endl;
	if( r==NULL )
		return false;

	//fprintf(stderr,"x=%f y=%f xmax=%f ymax=%f imgw=%f imgh=%f\n",(float)r->x,(float)r->y,(float)r->x+r->width,(float)r->y+r->height,(float)frame.cols(),(float)frame.rows());

	params->_initstate.resize(4);
	params->_initstate[0]	= (float)r->x;// - r->width;
	params->_initstate[1]	= (float)r->y;// - r->height;
	params->_initstate[2]	= (float)r->width;
	params->_initstate[3]	= (float)r->height+10;


	return true;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
				TrackerParams::TrackerParams()
{
	_boxcolor.resize(3);
	_boxcolor[0]	= 204;
	_boxcolor[1]	= 25;
	_boxcolor[2]	= 204;
	_lineWidth		= 2;
	_negnumtrain	= 15;
	_posradtrain	= 1;
	_posmaxtrain	= 100000;
	_init_negnumtrain = 1000;
	_init_postrainrad = 3;
	_initstate.resize(4);
	_debugv			= false;
	_useLogR		= true;
	_disp			= true;
	_initWithFace	= true;
	_vidsave		= "";
	_trsave			= "";
}

				SimpleTrackerParams::SimpleTrackerParams()
{
	_srchwinsz		= 30;
	_initstate.resize(4);
	_negsamplestrat	= 1;
}

