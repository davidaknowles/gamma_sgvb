function [l,g]=gpEpmLikelihood(par,XnotMissing,missing,mu)
        if nargin()==3
            mu=0;
        end
        % Pr(on)=1-exp(- W * diag(R) * W ) 
        % W_i: ~ G(a_i,c_i)
        % a_i ~ G(.01,.01)
        % c_i ~ G(1,1)
        % R_k ~ G( gamma0/K, c0 )
        % gamma0 ~ G(1,1) % equivalently gamma0/K ~ G(1,K)
        % c0 ~ G(1,1)
        N=size(XnotMissing,1); 
        K=size(par.W,2);
        
        aShape=.01; 
        aRate=.01; 

        W=par.W; 
        [ r, c ]=find(XnotMissing); 
        [ r2, c2 ]=find(missing); 
        r=[ r; r2 ]; 
        c=[ c; c2 ]; 
        ww=W(r,:) .* W(c,:); 
        temp1=sum( bsxfun(@times,ww,par.R'),2)+mu; 
        pred=sparse(r,c,temp1,N,N); 
        
        sW=sum(W,1); 
        % calculate gradient
        if nargout()>1
            tt=[ 1./(1-exp(-pred(XnotMissing))) ; ones(length(r2),1) ]; 
            tempg=sparse(r,c,tt,N,N)*W;
            g.W=bsxfun(@times,bsxfun(@plus,tempg,-sW),2.0*par.R') + ...
                bsxfun(@times,1 ./ W,par.a - 1.0)  - repmat( par.c, 1, K); 
            g.W( g.W < -1e10 )=-1e10; 
            g.a=sum(log(par.W),2)+K*log(par.c)-K*psi(par.a)+(aShape-1.0) ./ par.a - aRate; 
            g.a( g.a < -1e10 )=-1e10; 
            g.c=-sum(par.W,2)+K*par.a./par.c-1.0;
            g.c( g.c < -1e10 )=-1e10; 
            g.R=ww' * tt - sW' .^ 2 + ... %  diag( W' * sparse(r,c,tt) * W) + ... 
                 (par.gamma0dK - 1.0) ./ par.R - par.c0;
            g.R( g.R < -1e10 )=-1e10; 
            g.gamma0dK = sum(log(par.R)) + K*log(par.c0) - K*psi(par.gamma0dK) - K; % gamma(1,K) prior
            g.c0 = - sum(par.R) + K * par.gamma0dK ./ par.c0 - 1.0; 
            g.c0 = max(g.c0,-1e10); 
            g.gamma0dK = max(g.gamma0dK,-1e10); 
            assert( ~any(isinf(unwrap(g))));
        end
        
        % calculate likelihood
        temp=sparse(double(XnotMissing | missing));
        temp(XnotMissing)= log(1-exp(-pred(XnotMissing))) + pred(XnotMissing); 
        temp(missing)=pred(missing); 
        
        l=full( sum( temp(:) )) - sum(sW.^2 .* par.R') - N*N*mu + ...
             sum( (par.a - 1.0) .* sum(log(W),2) ) - sum( par.c .* sum(W,2))+ ...
             K * sum( par.a .* log(par.c) ) - K * sum(gammaln(par.a)) + ...
             (aShape - 1.0) * sum(log(par.a)) - aRate * sum(par.a) - sum(par.c) + ... 
                (par.gamma0dK - 1.0) .* sum(log(par.R)) - par.c0 * sum(par.R) + ...
                K * par.gamma0dK * log(par.c0) - K * gammaln( par.gamma0dK ) + ...
                - K * par.gamma0dK - par.c0;
         
    end