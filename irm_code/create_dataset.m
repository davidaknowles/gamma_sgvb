function [zz, nn, KK, R] = create_dataset(stat, param)
%%%%%%%%%%%%%%%%%%%%%% synthetic dataset construction %%%%%%%%%%%%%
% construct data using the real number of clusters


truez = ceil((1:stat.NN)/stat.NN*stat.trueK);

% define the real h matrix (matrix of link probabilities between clusters)
h = zeros(stat.trueK,stat.trueK);
for a=1:trueK
    for b=1:trueK
        if h(a,b)==0
            h(a,b)=betarnd(param.bbeta,param.bbeta);
            h(b,a)=h(a,b);
        end
    end
end


% based on the truez clustering of the persons and the h matrix of link probabilities among clusters, construct the
% real adjacency matrix
R = zeros(stat.NN,stat.NN);
for i=1:NN
    for j=(i+1):stat.NN
        
        a=truez(i);
        b=truez(j);
        R(i,j) = binornd(1,h(a,b));
        %R(i,j)=(a==b);
        R(j,i)=R(i,j);
        
    end
end

% aim : estimate the clustering z using as dataset the adjacency matrix R.
% will it be the same as the truez???


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KK=1;
% initialize clusters assignment
zz = ceil(rand(1,stat.NN)*KK); % all data initial put in one cluster (KK=1)
nn=zeros(1, KK); % it stores the number of persons assigned to each cluster
nn(1:KK)=stat.NN; % start with a single component containing all the persons
init=zz;

%Sanity check
%zz = ceil((1:NN)/NN*trueK);% initialize by putting the persons in the right clusters
%nn=zeros(1, trueK);
%for r=1:trueK
%    nn(r)=length(find(zz==r));
%end

end
