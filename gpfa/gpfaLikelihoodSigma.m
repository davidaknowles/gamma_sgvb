function [l,g]=gpfaLikelihoodSigma(par,yy,N,wshape,wrate,noisePrecShape,noisePrecRate)
    W=par.W; 
    D=size(W,1); 
    sigma2=1.0/par.noisePrec; 
    covar=W*W'+sigma2*eye(D); 
    ch=chol(covar); % prec=ch'*ch
    prec=ch\(ch'\eye(D)); 
    l=-N*sum(log(diag(ch)))-.5*sum( yy(:) .* prec(:) ); 
    l=l+(wshape-1) * sum(log(W(:))) - wrate * sum(W(:));
    l=l+noisePrecShape*log(par.noisePrec)-noisePrecRate*par.noisePrec;
    if nargout()>1
        precW=ch\(ch'\W); 
        g.W=-N*precW + ch\(ch'\(yy*precW))... % likelihood  
          + (wshape-1) ./ W - wrate; % prior
        pp=prec * prec; 
        g.noisePrec= .5 * ( N * sum(diag(prec)) - sum( yy(:) .* pp(:) ) )/par.noisePrec^2; 
    end
end