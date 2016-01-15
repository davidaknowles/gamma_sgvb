function [mu,sigma,ll,Xpred]=epmNormalSGVB(X,K,missing,settings)
    rng(1); 
    
    if nargin()==2
        missing=sparse(false(size(X))); 
    end
    
    N=size(X,1);

    mean(X(:))
    sum(X(:))
    
    XnotMissing=X & ~missing; 
    
    init.W=rand(N,K); 
    init.R=ones(K,1);
    init.c0=1;
    init.gamma0dK=1/K;
    init.a=ones(N,1)/K; 
    init.c=ones(N,1);
    likelihoodStruct=@(x) wrapFun( @(par) gpEpmLikelihood(par,XnotMissing,missing,1e-3), x, init ); 
    unconstrainedL=@(x) reparameteriser(likelihoodStruct, x);
    %initV=log(exp(unwrap(init))-1); 
    initV=randn(length(unwrap(init)),1);
    [mu,sigma,ll]=normalSGVB( unconstrainedL, initV, settings ); 
    mu=rewrap(init,mu); 
    sigma=rewrap(init,sigma);
    Xpred=getPred(mu,sigma);
    
    function Xpred=getPred(mu,sigma)
        samp=100; 
        Xpred=zeros(samp,N*N); 
        for ii=1:samp
            W=log1pe(mu.W+sigma.W.*randn(size(mu.W))); 
            R=log1pe(mu.R+sigma.R.*randn(size(mu.R))); 
            temp3=W*diag(R)*W';
            Xpred(ii,:)=1-exp(-temp3(:)); 
        end
        Xpred=mean(Xpred,1)'; 
    end

end