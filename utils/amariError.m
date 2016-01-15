function ae=amariError(EW,wtrue)

K=size(EW,2); 
assert(K==size(wtrue,2));
M=abs( (EW' * wtrue) / (wtrue' * wtrue) ); 
ae=( sum( sum(M,1) ./ max(M) - 1 ) +  sum( sum(M,2)' ./ max(M') - 1 ) ) / (2*K*K-2*K); 