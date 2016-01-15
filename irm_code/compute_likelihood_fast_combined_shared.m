function [lh lhtest  ] = compute_likelihood_fast_combined_shared(R, zz, param, settings )

N=size(R{1},1);

K=max(max(zz)); 
nlinks_tr=zeros(K,K); 
non_nlinks_tr=zeros(K,K); 
nlinks_test=zeros(K,K); 
non_nlinks_test=zeros(K,K); 

for r=1:length(R)

    i=zeros(N,K);

    for j=1:N
        i(j,zz(r,j)) = 1;
    end

    nlinks_tr=nlinks_tr+ i' * (R{r}.*settings.train_mask{r}) * i;
    non_nlinks_tr= non_nlinks_tr+ i' * ((1-R{r}).*settings.train_mask{r}) * i;

    nlinks_test=nlinks_test+ i' * (R{r}.*settings.test_mask{r}) * i;
    non_nlinks_test=nlinks_test+ i' * ((1-R{r}).*settings.test_mask{r}) * i;
end

beta1=param.bbeta1+(param.bbeta1_diag-param.bbeta1)*eye(K);
beta0=param.bbeta0+(param.bbeta0_diag-param.bbeta0)*eye(K);

lh=sum(sum(betaln(nlinks_tr+beta1,non_nlinks_tr+beta0)-betaln(beta1,beta0))); 

lhtest=sum(sum(betaln(nlinks_test+beta1,non_nlinks_test+beta0)-betaln(beta1,beta0))); 

assert(~isnan(lh) & ~isnan(lhtest)); 