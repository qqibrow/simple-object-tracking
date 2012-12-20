// MILTRACK
// Copyright 2009 Boris Babenko (bbabenko@cs.ucsd.edu | http://vision.ucsd.edu/~bbabenko).  Distributed under the terms of the GNU Lesser General Public License 
// (see the included gpl.txt and lgpl.txt files).  Use at own risk.  Please send me your feedback/suggestions/bugs.

#ifndef TRACKER_PUBLIC
#define TRACKER_PUBLIC

#include "OnlineBoost.h"
#include "Public.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class TrackerParams
{
public:
					TrackerParams();

	vectori			_boxcolor;						// for outputting video
	uint			_lineWidth;						// line width 
	uint			_negnumtrain,_init_negnumtrain; // # negative samples to use during training, and init
	float			_posradtrain,_init_postrainrad; // radius for gathering positive instances
	uint			_posmaxtrain;					// max # of pos to train with
	bool			_debugv;						// displays response map during tracking [kinda slow, but help in debugging]
	vectorf			_initstate;						// [x,y,scale,orientation] - note, scale and orientation currently not used
	bool			_useLogR;						// use log ratio instead of probabilities (tends to work much better)
	bool			_initWithFace;					// initialize with the OpenCV tracker rather than _initstate
	bool			_disp;							// display video with tracker state (colored box)

	string			_vidsave;						// filename - save video with tracking box
	string			_trsave;						// filename - save file containing the coordinates of the box (txt file with [x y width height] per row)

};



class SimpleTrackerParams : public TrackerParams
{
public:
					SimpleTrackerParams();

	uint			_srchwinsz;						// size of search window
	uint			_negsamplestrat;				// [0] all over image [1 - default] close to the search window
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class Tracker
{
public:
	
	static bool		initFace(TrackerParams* params, Matrixu &frame);
	static void		replayTracker(vector<Matrixu> &vid, string states, string outputvid="",uint R=255, uint G=0, uint B=0);
	static void		replayTrackers(vector<Matrixu> &vid, vector<string> statesfile, string outputvid, Matrixu colors);

protected:
	static CvHaarClassifierCascade	*facecascade;


};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class SimpleTracker : public Tracker
{
public:
					SimpleTracker(){ facecascade = NULL; };
					~SimpleTracker(){ if( _clf!=NULL ) delete _clf; };
	double			track_frame(Matrixu &frame, Matrixu &framedisp); // track object in a frame;  requires init() to have been called.
	double			PF_track_frame(Matrixu &frame, Matrixu &framedisp); //PF and Particle Filter tracker
	void			track_frames(vector<Matrixu> &video, SimpleTrackerParams p, ClfStrongParams *clfparams);  // initializes tracker and runs on all frames
	
	bool			init(Matrixu frame, SimpleTrackerParams p, ClfStrongParams *clfparams);
	Matrixf &		getFtrHist() { return _clf->_ftrHist; }; // only works if _clf->_storeFtrHistory is set to true.. mostly for debugging


private:
	ClfStrong			*_clf;
	vectorf				_curState;
	SimpleTrackerParams	_trparams;
	ClfStrongParams		*_clfparams;
	int					_cnt;


		////////  particlefilter  ///////////////
private:
    SampleSet particles;
};



#endif



