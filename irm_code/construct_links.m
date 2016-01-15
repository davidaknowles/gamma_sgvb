function [R,p] = construct_links(c,  h, param, settings)

N=length(c);
R = zeros(N,N);

% based on the true clustering of the persons and the h matrix of 
% link probabilities among clusters, construct the
% real adjacency matrix


p=h(c,c);
R =rand(N,N)<p;



end
