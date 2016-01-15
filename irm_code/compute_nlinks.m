function [mcheck mcheckt ] = compute_nlinks(R, zz, param )

% let's make it faster :)
ss = max(zz);


cl_links_tr = zeros(ss,ss);
cl_links_tr(1:end) = 1:1:(ss^2);
Rlabels = cl_links_tr(zz(:),zz(:));
Rtr=R;
Rtr(Rtr==0)=-1;
Rtr(param.test_mask==1)=0;
Rclean_tr = Rtr.*Rlabels;

[l_fr  el] =hist(Rclean_tr(1:end) , unique(Rclean_tr(1:end)));

% ss
% logical(el>0)
% keyboard
cl_links_tr(el(el>0))= l_fr(logical(el>0));
mcheck(:,:,1)=cl_links_tr;


cl_links_tr(abs(fliplr(el(el<0))))= fliplr(l_fr(logical(el<0)));
mcheck(:,:,2)=cl_links_tr;


%%%%%%%%

cl_links_tst = zeros(ss,ss);
cl_links_tst(1:end) = 1:1:(ss^2);
Rlabels = cl_links_tst(zz(:),zz(:));
Rtst=R;
Rtst(Rtst==0)=-1;
Rtst(param.train_mask==1)=0;
Rclean_tst = Rtst.*Rlabels;

[l_fr  el] =hist(Rclean_tst(1:end) , unique(Rclean_tst(1:end)));

cl_links_tst(el(el>0))= l_fr(logical(el>0));
mcheckt(:,:,1)=cl_links_tst;


cl_links_tst(abs(fliplr(el(el<0))))= fliplr(l_fr(logical(el<0)));
mcheckt(:,:,2)=cl_links_tst;

