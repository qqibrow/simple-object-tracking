// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#include "Matrix.h"
#include "ImageFtr.h"
#include "Tracker.h"
#include "Public.h"


void	Exp(const char* dir, const char* name, int strong, int trial, bool savevid=false);
void	FaceTrackDemo(const char* savefile);

int		main(int argc, char * argv[])
{
	string clipname;
	vector<string> clipnames;
	switch( 2 ){
		case 1: // FACE TRACK DEMO
			FaceTrackDemo(argc>1 ? argv[1] : 0);
			break;
		case 2: // RUN ON VIDEO
			if( argc<3 ) abortError(__LINE__,__FILE__,"Not enough parameters.  See README file.");
			Exp(argv[1],argv[2],(argc>3?atoi(argv[3]):1),(argc>4?atoi(argv[4]):1),(argc>5?atoi(argv[5])==1:0));
			break;					
		default:
			break;
	}

}
void	Exp(const char* dir, const char* name, int strong, int trial, bool savevid)
{
	bool success=true; 
	randinitalize(trial);

	string dataDir= "E://miltracker//" + string(dir);
	if( dataDir[dataDir.length()-2] != '//' ) dataDir+="//";
	dataDir += (string(name) + "//");

	Matrixf frameb, initstate;
	vector<Matrixu> vid;
	
	// read in frames and ground truth
	success=frameb.DLMRead((dataDir + name + "_frames.txt").c_str());
	if( !success ) abortError(__LINE__,__FILE__,"Error: frames file not found.");
	success=initstate.DLMRead((dataDir + name + "_gt.txt").c_str());
	if( !success ) abortError(__LINE__,__FILE__,"Error: gt file not found.");

	// TRACK

	vid.clear();
	vid = Matrixu::LoadVideo((dataDir+"imgs/").c_str(),"img", "png", (int)frameb(0), (int)frameb(1), 5, false);
	SimpleTracker tr;
	SimpleTrackerParams		trparams;
	vector<Matrixu>			saveseq;
	string paramname = "";
	

	////////////////////////////////////////////////////////////////////////////////////////////
	// PARAMETERS	

	ClfStrongParams			*clfparams;
	
	// strong model
	switch( strong ){
		case 1:		// MILTrack
			clfparams = new ClfMilBoostParams();
			((ClfMilBoostParams*)clfparams)->_numSel		= 50;
			((ClfMilBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_MIL";
			trparams._posradtrain							= 4.0f;
			trparams._negnumtrain							= 65;
			((ClfMilBoostParams*)clfparams)->ifPF = false;
			break;
		case 2:		// OBA1
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel		= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_OAB1";
			trparams._posradtrain							= 1.0f;
			trparams._negnumtrain							= 65;
			break;
		case 3:		// OBA5
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel		= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat		= 250;
			paramname += "_OAB5";
			trparams._posradtrain							= 4.0f;
			trparams._negnumtrain							= 65;
			break;
		case 4:		// PFMIL
		clfparams = new ClfMilBoostParams();
		((ClfMilBoostParams*)clfparams)->_numSel		= 50;
		((ClfMilBoostParams*)clfparams)->_numFeat		= 250;
		paramname += "_PFMIL";
		trparams._posradtrain							= 4.0f;
		trparams._negnumtrain							= 65;
		((ClfMilBoostParams*)clfparams)->ifPF = true;
		break;

		default:
			abortError(__LINE__,__FILE__,"Error: invalid classifier choice.");
	}

	// feature parameters
	FtrParams *ftrparams;
	HaarFtrParams haarparams;
	ftrparams = &haarparams;

	clfparams->_ftrParams	= ftrparams;

	// tracking parameters
	trparams._init_negnumtrain = 65;
	trparams._init_postrainrad = 3.0f;
	trparams._initstate[0]	= initstate(0,0);
	trparams._initstate[1]	= initstate(0,1);
	trparams._initstate[2]	= initstate(0,2);
	trparams._initstate[3]	= initstate(0,3);
	trparams._srchwinsz		= 25;
	trparams._negsamplestrat = 1;
	trparams._initWithFace	= false;

	trparams._debugv		= false;
	trparams._disp			= false; // set this to true if you want to see video output (though it slows things down)
	trparams._vidsave		= savevid? dataDir + name + paramname + "_TR" + int2str(trial,3) + ".avi" : "";
	trparams._trsave		= dataDir + name + paramname + "_TR" + int2str(trial,3) + "_c.txt";

	////////////////////////////////////////////////////////////////////////////////////////////
	// TRACK

	
	cout << "\n===============================================\nTRACKING: " << name + paramname + "_TR" + int2str(trial,3) << endl;
	cout <<   "-----------------------------------------------\n";
    
	
	//CvCapture* pCapture = cvCaptureFromFile("1.avi");

	tr.track_frames(vid, trparams, clfparams);
	cout << endl << endl;

	delete clfparams;
	
}


void	FaceTrackDemo(const char* savefile)
{
	float vwidth = 240, vheight = 180;  // images coming from webcam will be resized to these dimensions (smaller images = faster runtime)
	ClfStrongParams			*clfparams;
	SimpleTracker			tr;
	SimpleTrackerParams		trparams;

	////////////////////////////////////////////////////////////////////////////////////////////
	// PARAMS

	// strong model
	switch( 2 ){
		case 1:		// OBA1
			clfparams = new ClfAdaBoostParams();
			((ClfAdaBoostParams*)clfparams)->_numSel	= 50;
			((ClfAdaBoostParams*)clfparams)->_numFeat	= 250;
			trparams._posradtrain						= 1.0f;
			break;
		case 2:		// MILTrack
			clfparams = new ClfMilBoostParams();
			((ClfMilBoostParams*)clfparams)->_numSel	= 50;
			((ClfMilBoostParams*)clfparams)->_numFeat	= 250;
			trparams._posradtrain						= 4.0f;
			break;
		
	}

	// feature parameters
	FtrParams *ftrparams;
	HaarFtrParams haarparams;
	ftrparams = &haarparams;
	clfparams->_ftrParams	= ftrparams;

	// online boosting parameters
	clfparams->_ftrParams		= ftrparams;

	// tracking parameters
	trparams._init_negnumtrain	= 65;
	trparams._init_postrainrad	= 3.0f;
	trparams._srchwinsz			= 25;
	trparams._negnumtrain		= 65;


	// set up video
	CvCapture* capture = cvCaptureFromCAM( -1 );
	if( capture == NULL ){
		abortError(__LINE__,__FILE__,"Camera not found!");
		return;
	}


	////////////////////////////////////////////////////////////////////////////////////////////
	// TRACK

	// print usage
	fprintf(stderr,"MILTRACK FACE DEMO\n===============================\n");
	fprintf(stderr,"This demo uses the OpenCV face detector to initialize the tracker.\n");
	fprintf(stderr,"Commands:\n");
	fprintf(stderr,"\tPress 'q' to QUIT\n");
	fprintf(stderr,"\tPress 'r' to RE-INITIALIZE\n\n");
	
	// grab first frame
	Matrixu frame,framer,framedisp;
	Matrixu::CaptureImage(capture,framer);
	frame = framer.imResize(vheight,vwidth);

	// save output
	CvVideoWriter *w=NULL;

	// initialize with face location
	while( !Tracker::initFace(&trparams,frame) ){
		Matrixu::CaptureImage(capture,framer);
		frame = framer.imResize(vheight,vwidth);
		frame.display(1,2);
	}
	ftrparams->_height		= (uint)trparams._initstate[2];
	ftrparams->_width		= (uint)trparams._initstate[3];

	//这里初始化了
	tr.init(frame, trparams, clfparams);

	StopWatch sw(true);
	double ttime=0.0;
	double probtrack=0.0;

	// track
	for (int cnt = 0; Matrixu::CaptureImage(capture,framer); cnt++) {
		frame = framer.imResize(vheight,vwidth); 
		tr.track_frame(frame, framedisp);  // grab tracker confidence

		// initialize video output
		if( savefile != NULL && w==NULL ){
			w = cvCreateVideoWriter( savefile, CV_FOURCC('I','Y','U','V'), 15, cvSize(framedisp.cols(), framedisp.rows()), 3 );
		}

		// display and save frame
		framedisp._keepIpl=true;
		framedisp.display(1,2);
		if( w != NULL )
			Matrixu::WriteFrame(w, framedisp);
		framedisp._keepIpl=false; framedisp.freeIpl();
		char q=cvWaitKey(1);
		ttime = sw.Elapsed(true);
		fprintf(stderr,"%s%d Frames/%f sec = %f FPS, prob=%f",ERASELINE,cnt,ttime,((double)cnt)/ttime,probtrack);
	
		// quit
		if( q == 'q' )
			break;

		// restart with face detector
		else if( q == 'r' || probtrack<0 ) 
		{
			while( !Tracker::initFace(&trparams,frame) && q!='q' ){
				Matrixu::CaptureImage(capture,framer);
				frame = framer.imResize(vheight,vwidth);
				frame.display(1,2);
				q=cvWaitKey(1);
			}
			if( q == 'q' )
				break;
			
			// re-initialize tracker with new parameters
			ftrparams->_height		= (uint)trparams._initstate[2];
			ftrparams->_width		= (uint)trparams._initstate[3];
			clfparams->_ftrParams	= ftrparams;
			fprintf(stderr,"\n");
			tr.init(frame, trparams, clfparams);
			
		}
	}


	// clean up
	if( w != NULL )
		cvReleaseVideoWriter( &w );
	cvReleaseCapture( &capture );

}

