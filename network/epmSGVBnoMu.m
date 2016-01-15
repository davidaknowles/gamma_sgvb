function [a,b,ll,Xpred]=epmSGVBnoMu(X,K,missing,settings,idx_test)
    rng(1); 
    
    if nargin()==2
        missing=sparse(false(size(X))); 
    end
    
    N=size(X,1);

    mean(X(:))
    sum(X(:))
    
    XnotMissing=X & ~missing; 
    
    init.W=ones(N,K)/K; 
    init.R=ones(K,1);
    init.c0=1;
    init.gamma0dK=1/K;
    init.a=ones(N,1)/K; 
    init.c=ones(N,1);
    
    testErrRecord=[];
    doCallback=nargin()>=5; 
    [a,b,ll]=gammaSGVBwrap( @(par) gpEpmLikelihood(par,XnotMissing,missing), init, settings, @myCallback );
    
    Xpred=getPred(a,b); 

    function Xpred=getPred(a,b)
        samp=100; 
        Xpred=zeros(samp,N*N); 
        for ii=1:samp
            W=gamrnd(a.W,1./b.W);
            R=gamrnd(a.R,1./b.R);
            temp3=W*diag(R)*W';
            Xpred(ii,:)=1-exp(-temp3(:)); 
        end
        Xpred=mean(Xpred,1)'; 
    end

    function myCallback(a,b)
        if doCallback
            Xpred=getPred(a,b); 
            testerr=mean(abs(Xpred(idx_test)-X(idx_test))); 
            subplot(2,2,1);
            disp(['testerr: ',num2str(testerr)]); 
            testErrRecord(end+1)=testerr;
            hold off; loglog(1:length(testErrRecord),testErrRecord); 
            drawnow();
        end
    end


end