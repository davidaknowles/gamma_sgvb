function [a,b,ll]=gammaSGVBwrap( logL, init, settings, myCallback )

    settings.initalpha=unwrap(init);
    rewrapper=@(gg) rewrap(init,gg); 
    
    if nargin==3
        myCallback=@(a,b) 1; 
    end
    
    [a_unwrapped,b_unwrapped,ll]=gammaSGVB( @(x) wrapFun(logL, x, init), ...
        length(settings.initalpha), settings, myCallback );

    a=rewrapper(a_unwrapped); 
    b=rewrapper(b_unwrapped);

end