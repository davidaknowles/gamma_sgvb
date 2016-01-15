function [ zz, nn, KK ] = simple_gibbs(zz, nn, R, numiter, param, settings)

NN =length(zz);
KK = length(unique(zz));

ggamma=param.ggamma;
bbeta=param.bbeta;

for iter = 1:numiter
 
  for n = 1:NN
    kk = zz(n); % kk is current component that data item n belongs to
    nn(kk) = nn(kk) - 1; % subtract from number of data items in component kk
   
    % delete active component if it has become empty
    if nn(kk) == 0, 
      KK = KK - 1;
      nn(kk) = [];
      idx = find(zz>kk);
      zz(idx) = zz(idx) - 1;
    end
   
      l_v = zeros(1, KK+1);
   
      p= log([nn.*(1/(ggamma+NN-1))  ggamma*(1/(ggamma+NN-1)) ]);
      for k = 1:KK+1
        lh=compute_likelihood_fast(k, n, R, zz, param, settings);
        l_v(k) = lh;
        p(k) = p(k) + lh;
      end
      
    

      
      p = exp(p-max(p));
      p=p/(sum(p));
      
      
      
      % sample from the conditional probabilities
      %kk = find(mnrnd(1,p)==1);
      uu = rand;
      kk = 1+sum(uu>cumsum(p));
  
      if kk==KK+1
          nn(kk)=0;
          %zz(n)=kk;
          KK=KK+1;
      end
      
      zz(n) = kk; 
      nn(kk) = nn(kk) + 1; % increment number of data items in component kk
      
      
      
          
  end

end


end

