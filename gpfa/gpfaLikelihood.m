function [l,g]=gpfaLikelihood(w,yy,N,K,D,sigma2,wshape,wrate)
    W=reshape(w,[D K]); 
    covar=W*W'+sigma2*eye(D); 
    ch=chol(covar); % prec=ch'*ch
    prec=ch\(ch'\eye(D)); 
%     er=ch\y; 
    l=-N*sum(log(diag(ch)))-.5*sum( yy(:) .* prec(:) ); 
%     assert( sum(er(:).^2) == sum(diag(y'*(covar\y)))); 
%     assert( sum(er(:).^2) == sum( yy(:) .* prec(:) )); 
    if nargin()==5
        l=l-.5*wshape*(w'*w); % wshape=prior precision
    else
        l=l+(wshape-1) * sum(log(w)) - wrate * sum(w); 
    end
%    l=l-.5*(w'*w); 
    if nargout()>1
        precW=ch\(ch'\W); 
        g=-N*precW + ch\(ch'\(yy*precW)); 
        if nargin()==5
            g=g - wshape * W; 
        else
            g=g + (wshape-1) ./ W - wrate; 
        end
%         g=g - W; 
        g=g(:); 
    end
end