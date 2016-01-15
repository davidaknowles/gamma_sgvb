function [lh lhtest  ] = compute_likelihood_fast_combined(kk, n, R, zz, param, settings )

bbeta=param.bbeta; 

if kk>0
    zz(n)=kk;
 end
   
 
 
N=size(R{1},1);
K=length(unique(zz));
i=zeros(N,K);

for j=1:N
    i(j,zz(j)) = 1;
end

lh=0;
lhtest=0; 

for r=1:length(R)
    nlinks_tr= i' * (R{r}.*settings.train_mask{r}) * i;
    non_nlinks_tr= i' * ((1-R{r}).*settings.train_mask{r}) * i;

    nlinks_test= i' * (R{r}.*settings.test_mask{r}) * i;
    non_nlinks_test= i' * ((1-R{r}).*settings.test_mask{r}) * i;

    lh=lh+sum(sum(betaln(nlinks_tr+bbeta,non_nlinks_tr+bbeta)-betaln(bbeta,bbeta))); 

    lhtest=lhtest+sum(sum(betaln(nlinks_test+bbeta,non_nlinks_test+bbeta)-betaln(bbeta,bbeta))); 
    assert(~isnan(lh) & ~isnan(lhtest)); 
end
% 
%  [m mt] = compute_nlinks(R, zz, param )
% 
% bbeta=param.bbeta;
% 
%  %KK = length(unique(tzz));
%  %KK = max(tzz);
%  if kk>0
%     zz(n)=kk;
%  end
%     
%  NN = length(zz);
% 
% % m=mcheck;
% % mt=mchecktt;
% ss = max(zz);
% 
% temp =0;
% tempt=0;
% for ca=1:ss
%     for cb=1:ss
%         temp=temp+ betaln(m(ca,cb,1)+bbeta, m(ca,cb,2)+bbeta) - betaln(bbeta,bbeta);
%         tempt=tempt+ betaln(mt(ca,cb,1)+bbeta, mt(ca,cb,2)+bbeta) - betaln(bbeta,bbeta);
%     end
%  
% end
% 
% 
% lh =temp;
% lhtest=tempt;
% 
% end