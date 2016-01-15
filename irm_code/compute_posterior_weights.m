function [heta_vector] = compute_posterior_weights(zz_vector,  param)

 %find posterior weights heta *******************
 %recall that the inference algorithm does NOT compute the weights since
 %they are integrated out. In the following part, we recover them. 
    N=size(param.R,1);
    heta_vector=cell(1, length(zz_vector));
    for sample=1:length(zz_vector)
        
        zz=zz_vector{sample};
        
        K=max(zz); 
        i=sparse(1:N,zz,1,N,K);

        nlinks_tr= i' * param.Rtrain * i;
        non_nlinks_tr= i' * param.nRtrain * i;
        
        heta = betarnd(nlinks_tr+param.bbeta,non_nlinks_tr+param.bbeta);
        heta_vector{sample}=heta;
    end
