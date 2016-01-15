function [zz_launch, n_launch, KK_launch, prod_p]  = restricted_gibbs(zz, nn, R, i, j, iters, param, settings)

NN = length(zz);
KK_launch = length(unique(zz));
n_launch = nn;
zz_launch = zz;
ggamma=param.ggamma;
bbeta=param.bbeta;

z=zz;
z(i)=-1;
z(j)=-1;
% Store the set of observations that belong the same cluster(s) as i and
% j (excluding i and j)
S = find(z==zz(i) | z==zz(j));

%     b. Modify zz_launch by performing iter intermediate restricted Gibbs sampling scans.
for num =1:iters
    prod_p = 0;
    for t=1:length(S)
        p=S(t);
        kk = zz_launch(p); % kk is current component that data item n belongs to
        n_launch(kk) = n_launch(kk) - 1; % subtract from number of data items in component kk
        
        if (kk~=zz_launch(i) && kk~=zz_launch(j))
            pause
        end
        
        
        v = [n_launch(zz_launch(i)) n_launch(zz_launch(j))];
        pr= log(v.*(1/(ggamma+NN-1))); % this should be /sum(v) but it's a constant so it doesn't actually matter
        lh= [compute_likelihood_fast(zz_launch(i), p, R, zz_launch,param, settings)    compute_likelihood_fast(zz_launch(j), p, R, zz_launch,param, settings)];
        pr = pr+lh;
        
  
        
        pr = exp(pr-max(pr));
        pr=pr/(sum(pr));
        
        
        
        % sample from the conditional probabilities
        rd = find(mnrnd(1,pr)==1);
        %old =  zz_launch(p);
        switch rd
            case 1.0
                
                zz_launch(p) = zz_launch(i) ;
                n_launch(zz_launch(i)) = n_launch(zz_launch(i))+1;
                %if old~=zz_launch(p)
                prod_p = prod_p +log(pr(1));
                %end
            case 2.0
                
                zz_launch(p) = zz_launch(j) ;
                n_launch(zz_launch(j)) = n_launch(zz_launch(j))+1;
                %if old~=zz_launch(p)
                prod_p = prod_p + log(pr(2));
                %end
        end
        
        
        
    end
end




% disp('end...');


end

