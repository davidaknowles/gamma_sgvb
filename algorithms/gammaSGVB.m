% main function to perform Stochastic Gradient Variational Bayes with gamma
% variational posteriors. 
% logL - log likelihood function, second output arg is the gradient
% D - dimension of parameter space
% settings - various settings
% myCallback - can be used for plotting etc. 
function [a,b,ll]=gammaSGVB( logL, D, settings, myCallback )

eps=1e-10;

rng('shuffle');
tic

if isempty(settings.inita)
    a = settings.initalpha+zeros(D,1); 
    b = ones(D,1);
else
    a=settings.inita; 
    b=settings.initb;
end
% a(3)=10.0; warning('initialising a3=10!'); 

if settings.useMeanParameters % different parameterization, performed worse in general
    m=a./b; 
    v=a./b.^2; 
    r=[log(exp(m)-1); log(exp(v)-1)];
else
    % reparameterization to optimize in whole space
    r=[log(exp(a)-1); log(exp(b)-1)]; 
end
llCounter=0; % count likelihood evaluateion
ll=[]; 
if nargin()<=3
    ll=1;
    llCounter=1;
end

% test gradient evaluation using finite differences? useful for debugging
% but slow for big models
if settings.testGrad
    fprintf(1,'Testing gradients\n');
    tests=gamrnd(a,1./b); 
    [l0,ag]=logL(tests); 
%     ndeps=1e-6; 
    for d=1:D
       ndeps=min( tests(d)*1e-6, 1e-6); 
       temp=tests;
       temp(d)=temp(d)+ndeps; 
       nd= ( logL(temp) - l0 ) / ndeps; 
%        fprintf(1,'%f=%f\n',nd,ag(d));
       assert( abs(nd - ag(d) )/max(abs(nd),.001) < .05 ); 
    end
    fprintf(1,'Test grad passed\n');
end

lt=0;
% useMomentum=1; % set forgetting=1 for no momentum
% set up parameters for learning rate tuning methods
forgetting=.01; 
velocity=zeros(2*D,1); 
ms=zeros(D*2,1) * (1-settings.g2blend); 
if settings.useAdadelta
    msG=zeros(D*2,1);
    msX=zeros(D*2,1);
end
samplesUsed=zeros(settings.samples,1);
elboHistory=[]; 
for sampleIndex=1:settings.samples
    
    samplesUsed(sampleIndex)=samplesUsed(sampleIndex)+1; 
%         z=rand();
    minz=gamcdf(eps*ones(D,1),a,1./b); % don't allow z which give underflow
    z=minz+rand(D,1).*(1.0-minz);
    assert(all(~isnan(z))); % check for nan
    x=zeros(D,1);
    dfda=zeros(D,1); 
        
%        useApprox=7.8*log(a)+32*z < -22; 
   useApprox=(a<1) & ((24-22.6*z).*log(a) < -10);
   zua=z(useApprox); 
   aua=a(useApprox); 
   bua=b(useApprox); 
   logx=(log(zua)+log(aua)+gammaln(aua))./aua-log(bua);
   x(useApprox)=exp(logx);
   dlogxda=-(log(zua)+log(aua)+gammaln(aua))./aua.^2+(1./aua+psi(aua))./aua; 
   dfda(useApprox)=dlogxda .* x(useApprox);
   [x(~useApprox), dfda(~useApprox)]=mygaminv(z(~useApprox),a(~useApprox),b(~useApprox)); 
   dfdb=-x./b; 
           
    [logl,glp]=logL(x); 
    elboHistory(end+1)=logl +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) ); 
    if length(elboHistory)>10
       elboHistory(1)=[]; 
    end
    if settings.useAnalyticEntropy
        ghat= [ dfda .* glp + ( 1.0 + (1-a) .* psi(1,a) ) ;
                dfdb .* glp - 1.0./b  ] ; 
    else % seems slightly better
        glp=glp - (a-1)./x + b;
        ghat= [ dfda .* glp - log(x) - log(b) + psi(a) ;
                dfdb .* glp + x - a./b ] ; 
    end

    if settings.useMeanParameters
        ga=ghat(1:D); 
        gb=ghat((D+1):end); 
        ghat=[ (2*m.*ga+gb)./v ;
            - ( m.*ga+gb ).*m./v.^2 ]; 
    end
    
    ghat=ghat ./(1+exp(-r)); 
   if forgetting~=1.0
      velocity=(1-settings.forgetting)*velocity+settings.forgetting*ghat; 
      ghat=velocity;
   end

   if settings.useAdaGrad
       ms=settings.msBlend*ms+settings.g2blend*ghat.^2; 
       ghat=ghat ./ (1e-6 + sqrt(ms) );
   end

   if settings.useAdadelta
       msG=settings.rho*ghat.^2+(1-settings.rho)*msG;
       ghat = ghat .* sqrt( msX + settings.eps ) ./  sqrt( msG + settings.eps ); 
   end
       
   r=r+settings.stepSize*ghat;

   if settings.useAdadelta
       msX=settings.rho*ghat.^2+(1-settings.rho)*msX; 
   end

   if settings.useMeanParameters
       m=log1pe(r(1:D)); 
       v=log1pe(r((D+1):end));
       a=m.^2./v; 
       b=m./v;
   else
       a=log1pe(r(1:D)); 
       b=log1pe(r((D+1):end));
   end
   if mod(sampleIndex,20)==1
    llCounter=llCounter+1;
        ll(llCounter)=mean(elboHistory);
        if nargin()>3
            myCallback(a,b); 
        end
       if settings.plot
           subplot(2,2,4); 
           hold off; 
           plot(ll);  drawnow();
       end
   end
       
   minThres=1e-6; 

   tn=cputime(); 
   if tn-lt > 5
       fprintf(1,'%i, %f used approx, fixing %f shapes, %f rates, L=%g %g\n',...
           sampleIndex,mean(useApprox),mean(a<minThres),mean(b<minThres),ll(llCounter),mean(elboHistory));
       lt=tn; 
   end
       
   a(a<minThres)=minThres; 
   b(a<minThres)=minThres;

   assert( ~any(isinf(a)) & ~any(isinf(b)));
   assert( ~any(isnan(a)) & ~any(isnan(b) ));

end
toc


end