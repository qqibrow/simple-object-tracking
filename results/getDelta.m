function [x,delta,error] = getDelta(gtFileName, expFileName)

groundTruth = fopen(gtFileName);
expdata = fopen(expFileName);

[data,gtCount] = fscanf(expdata,'%f,%f,%f,%f\n');
gt = fscanf(groundTruth,'%f,%f,%f,%f\n');

 delta =[];
 x = [];
 
 for i = 1:20:gtCount
     dx = abs(data(i,1) - gt(i,1));
     dy = abs(data(i+1,1) - gt(i+1,1));
     
     
     delta = [delta dx+dy];
     x = [x (i-1)/4 + 1];
 end
 
 error = mean(delta);