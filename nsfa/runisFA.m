addpath('utils');
load example.mat
Ycentered=Y-repmat(mean(Y,2),1,size(Y,2));
settings=defaultsettings();
[settings.D,settings.N]=size(Ycentered); 
settings.iterations=100;
settings.verbose=1;
mvmask=binornd(1,1-0.1,settings.D,settings.N);
initialsample=initisFA(settings);
[finalsample,resultstable]=isFA(Ycentered,mvmask,initialsample,settings);
plot(resultstable(:,1),resultstable(:,2),'k+-'); 
xlabel('cpu time');
ylabel('log joint probability');