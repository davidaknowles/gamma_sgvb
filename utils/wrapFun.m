function [l,g]=wrapFun(myfun, x, s) % x is a vector, s is the struct
    h=rewrap(s,x); 
    if nargout()>1
        [l,gh]=myfun(h);
        g=unwrap(gh);
    else
        l=myfun(h);
    end
end