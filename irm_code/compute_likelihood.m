function [lh lhtest mlinks_tr mlinks_tst] = compute_likelihood(kk, n, R, zz, param, settings )
% COMPUTE_LIKELIHOOD
% R - adjacency matrix
% h - matrix with link prob among clusters
% zz - current cluster assignment

bbeta=param.bbeta;

 %KK = length(unique(tzz));
 %KK = max(tzz);
 if kk>0
    zz(n)=kk;
 end
    
 NN = length(zz);


% structure that holds the number of links and no_links 
% between each pair of clusters
%max(zz) == length(unique(zz))
ss = max(zz);
m=zeros(ss, ss, 2);


Rtrain=R;
Rtrain(settings.test_mask==1)=-1;

Rtest=R;
Rtest(settings.train_mask==1)=-1;

mcheck=zeros(ss, ss, 2); 
mchecktt=zeros(ss, ss, 2); 
for i=1:NN
	for j=1:NN
		mcheck(zz(i),zz(j),1)=mcheck(zz(i),zz(j),1)+(Rtrain(i,j)==1);
        mcheck(zz(i),zz(j),2)=mcheck(zz(i),zz(j),2)+(Rtrain(i,j)==0);
        
        	mchecktt(zz(i),zz(j),1)=mchecktt(zz(i),zz(j),1)+(Rtest(i,j)==1);
        mchecktt(zz(i),zz(j),2)=mchecktt(zz(i),zz(j),2)+(Rtest(i,j)==0);
	end
end

mlinks_tr=mcheck;
mlinks_tst=mchecktt;


% % let's make it faster :)
% Rlabels= R;
% Rlabels(:, :) =  
% 
% cl_links_tr = zeros(ss,ss);
% cl_links_tr(1:end) = 1:1:(ss^2);
% Rlabels = cl_links_tr(zz(:),zz(:));
% Rtr=R;
% Rtr(Rtr==0)=-1;
% Rtr(param.test_mask==1)=0;
% Rclean_tr = Rtr.*Rlabels;
% 
% [l_fr  el] =hist(Rclean_tr(1:end) , unique(Rclean_tr(1:end)));
% 
% % ss
% % logical(el>0)
% % keyboard
% cl_links_tr(el(el>0))= l_fr(logical(el>0));
% mcheck(:,:,1)=cl_links_tr;
% 
% 
% cl_links_tr(abs(fliplr(el(el<0))))= fliplr(l_fr(logical(el<0)));
% mcheck(:,:,2)=cl_links_tr;
% 
% 
% %%%%%%%%
% 
% cl_links_tst = zeros(ss,ss);
% cl_links_tst(1:end) = 1:1:(ss^2);
% Rlabels = cl_links_tst(zz(:),zz(:));
% Rtst=R;
% Rtst(Rtst==0)=-1;
% Rtst(param.train_mask==1)=0;
% Rclean_tst = Rtst.*Rlabels;
% 
% [l_fr  el] =hist(Rclean_tst(1:end) , unique(Rclean_tst(1:end)));
% 
% cl_links_tst(el(el>0))= l_fr(logical(el>0));
% mcheckt(:,:,1)=cl_links_tst;
% 
% 
% cl_links_tst(abs(fliplr(el(el<0))))= fliplr(l_fr(logical(el<0)));
% mcheckt(:,:,2)=cl_links_tst;
% 
% 
% 

% likelihood = P(R|z)
m=mcheck;
mt=mchecktt;

temp =0;
tempt=0;
for ca=1:ss
    for cb=1:ss
        temp=temp+ betaln(m(ca,cb,1)+bbeta, m(ca,cb,2)+bbeta) - betaln(bbeta,bbeta);
        tempt=tempt+ betaln(mt(ca,cb,1)+bbeta, mt(ca,cb,2)+bbeta) - betaln(bbeta,bbeta);
    end
 
end


lh =temp;
lhtest=tempt;



end
