function [lenergy] = compute_energy(zz, nn, KK, settings, param, R)

%energy=lk+prior(Z)
lh = compute_likelihood_fast(-1, -1, R, zz, param, settings );


temp = 0;
for i=1:KK
    temp  = temp + gammaln(nn(i));
end
lprior_zz = gammaln(param.ggamma)+KK*log(param.ggamma) +temp - gammaln(param.ggamma+settings.NN);

lenergy=lh+lprior_zz;

end

