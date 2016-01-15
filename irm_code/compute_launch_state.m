
function [zz_launch, n_launch, KK_launch] = compute_launch_state(zz, nn, i, j)

%**************function that defines the zz_launch state 

zz_launch = zz;
n_launch=nn;
KK_launch = length(unique(zz));


z=zz;
z(i)=-1;
z(j)=-1;
% Store the set of observations that belong the same cluster(s) as i and
% j (excluding i and j)
S = find(z==zz(i) | z==zz(j));

% If zz(i) == zz(j), then let zz_launch(i) be set to a new component and
% zz_launch(j)= zz(j)
if zz(i) == zz(j)
    temp = zz_launch(i);
    zz_launch(i) = KK_launch+1;
    n_launch(temp) = n_launch(temp)-1;
    KK_launch= KK_launch+1;
    n_launch(KK_launch)=1;
end




% For every k in S, set zz_launch(k) to either of the distinct components,
% zz_launch(i) or zz_launch(j) as follows:
%    a. Select an initial state by ramdonly setting, independently with
%    equal probability, zz_launch(k) to either zz_launch(i) or
%    zz_launch(j).
%
for t=1:length(S)
    k=S(t);
    coin = randi(2,1,1);
    temp  = zz_launch(k);
    switch coin
        case 1
            
            zz_launch(k) = zz_launch(i);
            
            n_launch(zz_launch(i)) = n_launch(zz_launch(i))+1;
        case 2
            
            zz_launch(k) = zz_launch(j);
            
            n_launch(zz_launch(j)) = n_launch(zz_launch(j))+1;
    end
    n_launch(temp) = n_launch(temp)-1;
    
    % no need to check if any of the zz(i) or zz(j) clusters has emptied
    % because there will always be the i and j in those.
end

