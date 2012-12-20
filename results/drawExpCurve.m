function avgError = drawExpCurve(videoName)

file_path = [videoName '/'];
cd(file_path);

avgError = zeros(1,4);
gtName = [videoName '_gt.txt'];
vid_PFMILname = [videoName '_PFMIL_TR001_c.txt'];
vid_MILname = [videoName '_MIL_TR001_c.txt'];
vid_Adaname = [videoName '_OAB1_TR001_c.txt'];
vid_Ada5name = [videoName '_OAB5_TR001_c.txt'];

[x,PFMILexpdata,avgError(1,1)] = getDelta(gtName,vid_PFMILname);
[x,MILexpdata,avgError(1,2)] = getDelta(gtName,vid_MILname);
[x,Adaexpdata,avgError(1,3)] = getDelta(gtName,vid_Adaname);
[x,Ada5expdata,avgError(1,4)] = getDelta(gtName,vid_Ada5name);



%red-----PFMIL
%green-----MILexpdata
%blue-----Adaboostexpdata
%yellow-----Ada5

 figure(2)
 plot(x,PFMILexpdata,'r',x,MILexpdata,'g',x,Adaexpdata,'b',x,Ada5expdata,'y')
 xlabel('frame number')
 ylabel('center location error(pixel)')
 title([videoName ' EXP results'])
 legend('red---PF&MIL(mine)','green---MIL','blue---Ada','blue---Ada5','Location','NorthWest');
%  save
 print(2,'-djpeg',[videoName '.jpeg']); 


writer = fopen(videoName,'w');
c = fprintf(writer,'%f\n',avgError);

cd('../');


 

