    % simplified non-hierarchical version of EPM for testing
    function [l,g]=networkLikelihood(W,XnotMissing,missing,sz,mu,priorShape,priorRate)
        W=reshape(W,sz); 
        N=sz(1);
        if 0 % slow version of likelihood calculation, only for debugging
            pred=W*W'+mu;
        else
            [ r, c ]=find(XnotMissing); 
            [ r2, c2 ]=find(missing); 
            r=[ r; r2 ]; 
            c=[ c; c2 ]; 
            temp1=sum(W(r,:) .* W(c,:),2) + mu; 
            pred=sparse(r,c,temp1); 
        end
        
%         temp=sparse(N,N); % slow
        
        % calculate gradient
        if nargout()>1
%             temp(XnotMissing)= 1./(1-exp(-pred(XnotMissing)));  % slow
            tt=1./(1-exp(-pred(XnotMissing))); 
            temp2=sparse(r,c, [ tt ; ones(length(r2),1) ] );
%             temp(missing)=1.0; % slow
            g=2.0*(temp2*W);
        %    g=2.0*(X./(1-exp(-pred)))*W; % assumes X is symmetric  slow
            g=bsxfun(@plus,g,-2.0*sum(W,1)) + ...
                (priorShape - 1.0) ./ W - priorRate;
            g=g(:);
        end
        
        % calculate likelihood
        temp(XnotMissing)= log(1-exp(-pred(XnotMissing))) + pred(XnotMissing); 
        temp(missing)=pred(missing); 
%         l=full( sum(sum(X .* (log(1-exp(-pred)) + pred))) ) - sum(sum( W,1 ).^2) - N*N*mu; % slow
        l=full( sum( temp(:) )) - sum(sum( W, 1 ).^2) - N*N*mu + ...
             (priorShape - 1.0) .* sum(log(W(:))) - priorRate * sum(W(:)); 
%         if 0 % debugging
%             lcheck=sum(sum( (1-missing) .* (X .* log(1-exp(-pred)) + (1-X) .* (-pred) ))) + ...
%              (priorShape - 1.0) .* sum(log(W(:))) - priorRate * sum(W(:)); 
%             assert( abs(lcheck-l)<.01 );
%         end

    end