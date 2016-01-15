function [x,ll]=gradientAscentSimple(myfun,x,settings)
    if settings.testGrad
       testgrad( myfun, x); 
    end
%     hist(x); drawnow();
    for i=1:settings.samples
        [ll(i),ghat]=myfun(x); 
        fprintf(1,'%i %f\n',i,ll(i));
        x=x+settings.stepSize * ghat; 
        if i>1 & (abs(ll(i)-ll(i-1))<1e-8)
            break
        end
%         figure();
%         hist(x); drawnow();
%         testgrad( myfun, x); 
    end