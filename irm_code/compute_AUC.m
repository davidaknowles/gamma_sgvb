% Konstantina Palla <kp376@cam.ac.uk>
% University of Cambridge
% Created: February 2012
% Modified: August 2012

function [auc] = compute_AUC(param, settings, err)
% Function that compute the Area Under the Curve estimate of the model

%     bern_vec: a *vector* that contains the bernulli probabilities predicted 
%               by the model for each pair of objects. Notice that if the data is NxN 
%               the bern_vec is a vector and not a NxN matrix because you want ONLY the TEST data which is smaller in size. 
%     R_vec: a vector of the corresponding entries in the real link matrix (ground truth). A vector of zeros and ones. 
% 

idxs=find(settings.test_mask==1);
R_vec = param.R(idxs);
p_vec=err.bern_prob(idxs);

[ranked_bern rank_indcs] = sort(p_vec);
R_sorted = R_vec(rank_indcs);

% AUC computation
% A = (So - nl*(nl+1)/2) / (nl*nn)

%nl, nn are the numbers of links and non-links (ones and zeros) in the true link
%matrix respectively
nl = sum(R_sorted>0);
nn = sum(R_sorted<1);

%So = \sum(ri) where ri is the rank of the ith positive example (one -
%link) in the ranked list

So = sum(find(R_sorted > 0));

auc = (So - (nl*(nl+1))/2)/(nl*nn);
    
end