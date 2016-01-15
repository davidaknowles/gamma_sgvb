function gg = slice_sample_gamma(param, zz, nn, KK)

gg=param.ggamma;
%p(ggama | z) prop_to p(z|ggama)*p(ggama)
NN= length(zz);

temp = 0;
for i=1:KK
    temp  = temp + gammaln(nn(i));
end
f= @(g) gammaln(g)+KK*log(g) +temp - gammaln(g+NN);
h =@(x) f(exp(x))+x-exp(x);
N = 1;
t= slicesample(log(gg),N,'logpdf',h,'thin',1,'burnin',0);
gg=exp(t);
end
