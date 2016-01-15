function [x,g]=mygaminv(p,a,rates)
    persistent xx yy v
    if isempty(xx) 
%         [xx yy]=meshgrid( 0:0.001:.999, exp(-7:.01:7)); 
        [xx yy]=meshgrid( 0:0.01:.99, exp(-7.2:.1:7)); 
        v=gaminv( xx, yy); 
    end
    if numel(a)>0
        eps=1e-10;
        x=zeros(length(p),1); 
        g=zeros(length(p),1); 

        assert(all(a>exp(-7.2))); 

        useNorm=a>1000; 
        z=norminv(p(useNorm)); % could just ignore p and use randn
        sqa=sqrt(a(useNorm));
        x(useNorm)=a(useNorm)+sqa.*z; 
        g(useNorm)=1+.5.*z./sqa;

        useFull=(~useNorm) & ( ( (a>1) & (p<0.01 | p>0.99) ) | (a<=1 & p>.99) ) ; 
        x(useFull) = gammaincinv(p(useFull),a(useFull));
        g(useFull) = ( gammaincinv(p(useFull),a(useFull)+eps) - x(useFull)) ./ eps;

        useInterp=(~useNorm) & (~useFull);
        x(useInterp) =interp2(xx,yy,v,p(useInterp),a(useInterp)); 
        g(useInterp) = ( interp2(xx,yy,v,p(useInterp),a(useInterp)+eps) - x(useInterp) ) ./ eps; 

        x=x./rates; 
        g=g./rates;
    else
        x=[]; 
        g=[];
    end
end
