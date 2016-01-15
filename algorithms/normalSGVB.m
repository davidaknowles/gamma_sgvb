function [mu,sigma,ll]=normalSGVB( logL, mu, settings )

rng('shuffle');
tic
D=length(mu);
sigma = ones(D,1);
% a(3)=10.0; warning('initialising a3=10!'); 

r=[mu; log(exp(sigma)-1)];
llCounter=0; 
ll=[]; 
if nargin()<=3
    ll=1;
    llCounter=1;
end

if settings.testGrad
    fprintf(1,'Testing gradients\n');
    tests=mu+sigma .* randn(D,1); 
    [l0,ag]=logL(tests); 
    ndeps=1e-6; 
    for d=1:D
       temp=tests;
       temp(d)=temp(d)+ndeps; 
       nd= ( logL(temp) - l0 ) / ndeps; 
%        fprintf(1,'%f=%f\n',nd,ag(d));
       assert( abs(nd - ag(d) )/max(abs(nd),.001) < .01 ); 
    end
    fprintf(1,'Test grad passed\n');
end

% useMomentum=1; % set forgetting=1 for no momentum
velocity=zeros(2*D,1); 
ms=zeros(D*2,1) * (1-settings.g2blend); 
if settings.useAdadelta
    msG=zeros(D*2,1);
    msX=zeros(D*2,1);
end
samplesUsed=zeros(settings.samples,1); 
elboHistory=[]; 
lt=0;
for sampleIndex=1:settings.samples
    
    samplesUsed(sampleIndex)=samplesUsed(sampleIndex)+1; 
	z=randn(D,1); 
    x=mu+sigma .* z; 
    
    dfdm=ones(D,1); 
    dfdsigma=z; 
    
       [logl,glp]=logL(x); 
       elboHistory(end+1)=logl + .5 * (D + D *log(2*pi)) + sum(log(sigma)); 
       if length(elboHistory)>10
           elboHistory(1)=[]; 
       end
    if settings.useAnalyticEntropy % seems roughly the same
        ghat= [ dfdm .* glp ;
                dfdsigma .* glp + 1.0./sigma   ]  ; 
    else
        glp=glp+(x-mu)./(sigma.^2); 
        ghat= [ dfdm .* glp - (x-mu)./(sigma.^2) ; % TODO redundant
            dfdsigma .* glp - (x-mu).^2./sigma.^3 + 1.0./sigma   ]  ; 
    end
    
   ghat( (D+1):end ) = ghat( (D+1):end ) ./ (1+exp(-r((D+1):end))); 

   if settings.forgetting~=1.0
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

   mu=r(1:D); 
   sigma=log1pe(r((D+1):end));
       
   if mod(sampleIndex,20)==1
       llCounter=llCounter+1;
        ll(llCounter)=mean(elboHistory);
       if settings.plot
           subplot(2,2,4); plot(ll); drawnow();
       end
   end

   

          
   tn=cputime(); 
   if tn-lt > 5
        fprintf(1,'%i, L=%g\n',sampleIndex,ll(llCounter));
       lt=tn; 
   end

   assert( ~any(isinf(mu)) & ~any(isinf(sigma)));
   assert( ~any(isnan(mu)) & ~any(isnan(sigma) ));


end
toc
    
    function lp=log1pe(in)
        lp=zeros(size(in)); 
        isLinear=in>10; 
        lp(isLinear)=in(isLinear); 
        lp(~isLinear)=log1p(exp(in(~isLinear)));
    end

end