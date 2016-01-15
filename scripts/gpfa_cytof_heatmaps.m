load('data/cytof.mat'); 

y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y,[],2));

N=size(y,2); 
D=size(y,1); 
K=10;

settings=gammaSGVBsettings(6); 
settings.plot=0;
settings.testGrad=0;
settings.samples=5000;

[a,b,sgvbCov]=gpfaSigma(y,K,settings);
EW=a.W ./ b.W; 
cg=clustergram(EW); 
[~,permrows] = ismember(get(cg,'RowLabels'),num2str( (1:39)'))
co=cov(y'); 

subplot(1,3,1); imagesc( co(permrows,permrows), [0 1] );  
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
subplot(1,3,2); imagesc( sgvbCov(permrows,permrows), [0 1] ); 
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
[~,ordlatent]=sort(-sum(EW,1));
subplot(1,3,3); imagesc( EW(permrows,ordlatent), [0 1] ); colormap('default'); xlabel('latent factors'); 
set(gca,'YTickLabel',[]);
colorbar();