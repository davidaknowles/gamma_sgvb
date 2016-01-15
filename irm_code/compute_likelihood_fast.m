function [lh lhtest  ] = compute_likelihood_fast(kk, n, R, zz, param, settings )

if iscell(R)
   [lh lhtest]= compute_likelihood_fast_combined(kk, n, R, zz, param, settings );
   return
end

bbeta=param.bbeta; 

if kk>0
    zz(n)=kk;
end
   
N=size(R,1);
% K=length(unique(zz));
K=max(zz); 

%i=zeros(N,K); 
%i( sub2ind( [N K], 1:N, zz ) ) =1; 

i=sparse(1:N,zz,1,N,K);

% if 0
%     islow=zeros(N,K);
%     for j=1:N
%         islow(j,zz(j)) = 1;
%     end
%     assert( isequal(i,islow));
% end

% nlinks_tr= i' * (R.*settings.train_mask) * i;
% non_nlinks_tr= i' * ((1-R).*settings.train_mask) * i;

nlinks_tr= i' * param.Rtrain * i;
non_nlinks_tr= i' * param.nRtrain * i;

lh=sum(sum(betaln(nlinks_tr+bbeta,non_nlinks_tr+bbeta)-betaln(bbeta,bbeta))); 
assert(~isnan(lh));

if nargout()>1
    nlinks_test= i' * (R.*settings.test_mask) * i;
    non_nlinks_test= i' * ((1-R).*settings.test_mask) * i;
    lhtest=sum(sum(betaln(nlinks_test+bbeta,non_nlinks_test+bbeta)-betaln(bbeta,bbeta))); 
    assert(~isnan(lhtest)); 
end

return 
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