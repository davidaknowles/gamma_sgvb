function [par,ll,Xpred]=epmMAP(X,K,missing,settings,idx_test)
    rng(1); 
    
    if nargin()==2
        missing=sparse(false(size(X))); 
    end
    
    N=size(X,1);

    mean(X(:))
    sum(X(:))
    
    XnotMissing=X & ~missing; 
    
    init.W=rand(N,K)/K; 
    init.R=ones(K,1);
    init.c0=1;
    init.gamma0dK=1/K;
    init.a=ones(N,1)/K; 
    init.c=ones(N,1);
    likelihoodStruct=@(x) wrapFun( @(par) gpEpmLikelihood(par,XnotMissing,missing), x, init ); 
    unconstrainedL=@(x) reparameteriser(likelihoodStruct, x);
    testErrRecord=[];
    doCallback=nargin()>=5;
    [x, ll, ~] = minimize( log(exp(unwrap(init))-1), @(x)signflipper(unconstrainedL,x), -1000, @myCallback ); 
    par=rewrap(init,log1pe(x)); 
    Xpred=getPred(par);
    
    function Xpred=getPred(par)
        temp2=par.W*diag(par.R)*par.W';
        Xpred=1-exp(-temp2(:));
    end
    
    function myCallback(x)
        if doCallback
            par=rewrap(init,log1pe(x)); 
            Xpred=getPred(par);
            testerr=mean(abs(Xpred(idx_test)-X(idx_test))); 
            disp(['testerr: ',num2str(testerr)]); 
            testErrRecord(end+1)=testerr;
            hold off; loglog(1:length(testErrRecord),testErrRecord); 
            drawnow();
        end
    end

    function [l,g]=signflipper(myfun,x)
        if nargout()==2
            [l,g]=myfun(x); 
            l=-l; 
            g=-g;
        else
            l=myfun(x);
        end
    end

end