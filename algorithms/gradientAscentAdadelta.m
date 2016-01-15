function [x,ll]=gradientAscentAdadelta(myfun,x,settings)

    D=numel(x);
    
    if settings.testGrad
       testgrad( myfun, randn(length(x),1)); 
    end
    msG=zeros(D,1);
    msX=zeros(D,1);
    for i=1:settings.samples
        [ll(i),ghat]=myfun(x); 
        fprintf(1,'%i %f\n',i,ll(i));
        plot(ll); drawnow();
        msG=settings.rho*ghat.^2+(1-settings.rho)*msG;
        ghat = ghat .* sqrt( msX + settings.eps ) ./  sqrt( msG + settings.eps ); 
        msX=settings.rho*ghat.^2+(1-settings.rho)*msX;
        x=x+settings.stepSize * ghat; 
        if i>1 & (abs(ll(i)-ll(i-1))<1e-8)
            break
        end
    end