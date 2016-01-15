function bb = slice_sample_beta(R, param, zz, nn, KK)

bb=param.bbeta;
NN = length(zz);
ss = max(zz);
m=zeros(ss, ss, 2);
mcheck=zeros(ss, ss, 2); 
for i=1:NN
	for j=1:NN
		mcheck(zz(i),zz(j),1)=mcheck(zz(i),zz(j),1)+(R(i,j)==1);
        mcheck(zz(i),zz(j),2)=mcheck(zz(i),zz(j),2)+(R(i,j)==0);
	end
end
% likelihood = P(R|z)
m=mcheck;
temp =0;

lh = @(bbeta) likelihood(bbeta,m); 
f = @(bbeta) lh(bbeta) -2.5*log(bbeta);

%h =@(x) f(exp(x))+x;


N = 1;
t= slicesample(bb,N,'logpdf',f,'thin',1,'burnin',0);

bb=t;
%bb=exp(t);

    function l=likelihood(bbeta,m)
        l=0; 
        for ca=1:ss
            for cb=1:ss
                l= l+ betaln(m(ca,cb,1)+bbeta, m(ca,cb,2)+bbeta) - betaln(bbeta,bbeta);
            end
        end
 
    end

end
