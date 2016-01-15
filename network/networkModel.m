% use gammaSGVB to fit the EPM
function [par,ll,map,Xpred]=networkModel(X,K,missing,settings)

    mu=1e-5; 
        
    if nargin()==2 % assume all observed by default
        missing=sparse(false(size(X))); 
    end
    
    priorShape=1/K; priorRate=1; 
    N=size(X,1);

    sz=[N K];
    D=N*K; % total parameters in model
    
    XnotMissing=X & ~missing; 
    logL=@(W) networkLikelihood(W,XnotMissing,missing,sz,mu,priorShape,priorRate); 

    map=[];
    [a,b,ll]=gammaSGVB( logL, D, settings );
    
    par.mu=mu; 
    par.Wshape=reshape(a,sz); 
    par.Wrate=reshape(b,sz); 

    samp=30; 
    Xpred=zeros(samp,N*N); 
    for i=1:samp
        W=gamrnd(par.Wshape,1./par.Wrate);
        mu=par.mu;
        temp=W*W'+mu; 
        Xpred(i,:)=1-exp(-temp(:));
    end
    Xpred=mean(Xpred,1)';

end