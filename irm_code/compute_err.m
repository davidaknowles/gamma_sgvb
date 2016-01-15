function [st] = compute_err( Zs_iter, hs_iter,  Rtrue, settings , param)

test_count = sum( settings.test_mask(:) );
train_count = sum( settings.train_mask(:) );
N=size(Rtrue,1);
bern_v=zeros(N,N);
% compute for several samples
train_err=zeros(1,settings.evaluation_window);
test_err=zeros(1,settings.evaluation_window);
lkl_trn=zeros(1,settings.evaluation_window);
lkl_tst=zeros(1,settings.evaluation_window);

for i = 1:settings.evaluation_window
    
    Rconstr=construct_links(Zs_iter{end - i + 1},  hs_iter{end - i + 1}, param, settings );
    err = ( Rtrue -  Rconstr).^2;
    
    train_err(i) = sum(sum( settings.train_mask .* err )) / train_count;
    test_err(i) = sum(sum(settings.test_mask .* err )) / test_count;
    
    [ lkl_trn(i) lkl_tst(i) ] = compute_likelihood_fast( -1, -1, Rtrue, Zs_iter{end-i+1}, param, settings );
    
    
    c=Zs_iter{end - i + 1};
    h=hs_iter{end - i + 1};
    bern=zeros(N,N);
    for i=1:N
        for j=1:N
            
            a=c(i);
            b=c(j);
            bern(i,j)= h(a,b);
            
        end
    end
    
    bern_v=bern_v+bern;
end

st.bern_prob=bern_v./settings.evaluation_window;


%%create R from the trained model
st.Rreconstructed=zeros(N,N);
for i=1:N
    for j=1:N
        st.Rreconstructed (i,j) = binornd(1,st.bern_prob(i,j));
    end
end
%OR
% st.Rreconstructed = rand(N,N)<st.bern_prob;


% average
st.train_err = mean( train_err );
st.test_err = mean( test_err );
st.test_count=test_count;
st.train_count=train_count;
st.ll_test_vector=lkl_tst;
st.ll_test=log(mean(exp(lkl_tst-max(lkl_tst))))+max(lkl_tst);
end





