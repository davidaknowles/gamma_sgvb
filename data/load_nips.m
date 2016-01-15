function [R, masks]=load_nips(holdouttype)

load nips_small.mat
N=size(R,1);

if holdouttype==0
    hold_out_per=0.2;
    holdout=ceil(hold_out_per*(N^2));
    per=randperm(N^2);
    test_mask = sparse( zeros(size(R)));
    test_mask(per(1:holdout))=1;
else
    nodes_to_test=rand(N,1)<0.1; 
    a=nodes_to_test * ones(1,N); 
    b=ones(N,1) * nodes_to_test'; 
    test_mask=(a | b) & (rand(N,N) < .9); 
end

test_mask=triu(test_mask);
test_mask=test_mask+test_mask'- diag(diag(test_mask));
train_mask=1-test_mask;

train_mask=train_mask-diag(diag(train_mask)); 
test_mask=test_mask-diag(diag(test_mask));

masks.test_mask = test_mask;
masks.train_mask =train_mask;

assert( isequal(train_mask,train_mask.') && isequal(test_mask,test_mask.') ); 

end
