function [l,g]=reparameteriser(myfun, x)
    h=log1pe(x); 
    if nargout()>1
        [l,gh]=myfun(h);
        g=gh ./ (1+exp(-x));
    else
        l=myfun(h);
    end
end