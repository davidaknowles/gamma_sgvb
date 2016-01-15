function [l,g]=gpfaLikelihoodSigmaHier(par,yy,N,noisePrecShape,noisePrecRate)
% model: 
% v_k ~ G( a0/K , rate=a0 )
% w_dk ~ G( a v_k, rate=a ) 
    a=1; 
    a0=1; 
    W=par.W; 
    D=size(W,1); 
    K=length(par.v); 
    
    sigma2=1.0/par.noisePrec; 
    covar=W*W'+sigma2*eye(D); 
    ch=chol(covar); % prec=ch'*ch
    prec=ch\(ch'\eye(D)); 
    l=-N*sum(log(diag(ch)))-.5*sum( yy(:) .* prec(:) ); 
    l=l+dot(a * par.v - 1 , sum(log(W),1)) - a * sum(W(:)) ...
        + D * a * sum(par.v) * log(a) - D * sum(gammaln(a * par.v)); 
    l=l+ (a0/K-1)*sum(log(par.v)) - a0 * sum(par.v) ...
        + a0 * log(a0) - K * gammaln(a0/K); 
    l=l+noisePrecShape*log(par.noisePrec)-noisePrecRate*par.noisePrec;
    if nargout()>1
        g.v=a * sum(log(W),1)' + D*a*log(a) - D*a*psi(a*par.v) ... % likelihood
            + (a0/K-1)./par.v - a0; 
        precW=ch\(ch'\W); 
        g.W=-N*precW + ch\(ch'\(yy*precW))... % likelihood  
          + bsxfun(@times,1./ W, a*par.v'-1) - a; % prior
        pp=prec * prec; 
        g.noisePrec= .5 * ( N * sum(diag(prec)) - sum( yy(:) .* pp(:) ) )/par.noisePrec^2; 
    end
end