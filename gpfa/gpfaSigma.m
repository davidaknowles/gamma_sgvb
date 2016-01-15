function [a,b,sgvbCov]=gpfaSigma(y,K, settings)
    
    D=size(y,1);
    N=size(y,2); 
    
    par.W=ones(D,K); % .1+rand(D,K); 
    par.noisePrec=1;
    par.v=ones(K,1)/K; %.1+rand(K,1);
    
    yy=y*y';
    % myfun=@(w) gpfaLikelihoodSigma(w,yy,N,.1,1,.1,.1); 
    myfun=@(w) gpfaLikelihoodSigmaHier(w,yy,N,.1,.1)
    [a,b,~]=gammaSGVBwrap( myfun, par, settings); 
    % EW=reshape(a.W ./ b.W ,[D K]);
    ecov=zeros(D,D); 
    nsamples=100;
    for i=1:nsamples
        W= gamrnd(a.W , 1./b.W );
        sigma2 = 1 / gamrnd(a.noisePrec, 1 ./ b.noisePrec);
        ecov=ecov+ W * W' + eye(D) * sigma2;
    end
    sgvbCov=ecov/nsamples; 
end