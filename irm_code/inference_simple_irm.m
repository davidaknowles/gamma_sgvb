function [zz_cell, lenergy_vector, g_vector, b_vector, param, settings] = inference_simple_irm(settings, param, R)

if iscell(R)
    settings.NN=size(R{1},1);
else
    settings.NN=size(R,1); 
end

KK=1;
% initialize clusters assignment
zz = ceil(rand(1,settings.NN)*KK); % all data initial put in one cluster (KK=1)

nn=zeros(1, KK); % it stores the number of persons assigned to each cluster
nn(1:KK)=settings.NN; % start with a single component containing all the persons

% vectors to store the gamma and beta values
g_vector = zeros(1,settings.numiter);
b_vector = zeros(1,settings.numiter);
%vector to store the number of clusters during each iteration
kk_vector = zeros(1,settings.numiter);
%vector to store the cluster assignments (labels) dusring each iteration
zz_cell= cell(1,settings.numiter);
%vector to store the energy
lenergy_vector = zeros(1,settings.numiter);

param.Rpred_iter = cell(1,settings.numiter);


param.Rtrain=R.*settings.train_mask;
param.nRtrain=(1-R).*settings.train_mask;


for iter = 1:settings.numiter
    fprintf('Iteration: %d \n', iter);
    
    %Start split merge procedure
    %Select two observations, i, j, at random uniformly
    i=0;
    j=0;
    while i==j
        v = randi(length(zz),1,2);
        i=v(1);
        j=v(2);
    end
    
    
    
    z=zz;
    z(i)=-1;
    z(j)=-1;
    % Store the set of observations that belong the same cluster(s) as i and
    % j (excluding i and j)
    S = find(z==zz(i) | z==zz(j));
    
    % compute the launch state
    [zz_launch, n_launch, KK_launch] = compute_launch_state(zz, nn, i, j);
    %************************************************
    
    % run intermediate restricted Gibbs
    %fprintf('Running %d restricted Gibbs scans...\n',settings.numiter_rstr_gibbs);
    [zz_launch n_launch KK_launch prod_p]= restricted_gibbs(zz_launch, n_launch, R, i, j, settings.numiter_rstr_gibbs, param, settings);
    
    
    % If items i and j are in the same mixture components, zz(i)==zz(j),
    % then
    % a. propose a new assignment of data items to mixture components,
    % denoted as zz_split, in which component zz(i) = zz(j) is split into
    % two separate components, zz_split(i) and zz_split(j). Define each
    % element of the proposal vector, zz_split, as follows:
    %
    %        .Let zz_split(i) = zz_launch(i)
    %        .Let zz_split(j) = zz_launch(j) (=zz(j))
    %        .For every observation p in S, let zz_split(k) be set either
    %         to component zz_split(i) or zz_split(j) by conducting one
    %         final Gibbs sampling scan from the launch state, zz_launch
    %        .For observation p not in S+{i,j}, let zz_split(k) = zz(k).
    
    state='';
    if zz(i) ==zz(j)
        %disp('i and j in the same cluster');
        %zz_split = zz_launch;% useless?
        %zz_split(i) = zz_launch(i);
        %zz_split(j) = zz_launch(j);
        
        [zz_split n_split KK_split prod_p]= restricted_gibbs(zz_launch, n_launch, R, i, j, 1, param, settings);
        
        % b. Calculate the proposal probability q(zz_split|zz) by computing
        % Gibbs sampling transition prob. from the launch state zz_launch
        % to the final proposed state, zz_split. The Gibbs sampling
        % transition prob. is the product, over p in S, of the
        % probabilities of setting each zz_split(p) to its final Gibbs
        % sampling scan.
        
        q_split_zz = prod_p;
        
        % evaluate the proposal by the MH acceptance probability
        % alpha(zz_split, zz)
        prior_ratio = log(param.ggamma) + gammaln(n_split(zz_split(i))) + gammaln(n_split(zz_split(j))) - gammaln(nn(zz(i)));
        
        % likelihood ratio
        % L(zz_split|R) / L(zz|R) = P(R|zz_split) / P(R|zz)
        likelihood_ratio = compute_likelihood_fast(zz_split(i), i, R, zz_split, param, settings) - compute_likelihood_fast(zz(i), i, R, zz, param, settings);
        
        % proposal ratio
        % q_ratio_split = q(zz|zz_split) / q(zz_split|zz)
        % q(zz|zz_split) = 1
        q_ratio = log(1) - q_split_zz;
        
        zz_new = zz_split;
        nn_new = n_split;
        KK_new = KK_split;
        state='split';
    else
        %zz(i) ~=zz(j)
        %fprintf('i and j not in the same cluster...\n');
        
        zz_merge = zz;
        n_merge = nn;
        KK_merge = KK;
        
        
        %a. for every observation p in S, let zz_merge= zz(j)
        
        zz_merge(i) = zz(j);
        zz_merge(S) = zz(j);
        n_merge(zz(j)) = n_merge(zz(j))+nn(zz(i));
        n_merge(zz(i)) = [];
        
        KK_merge = KK_merge - 1;
        idx = find(zz_merge>zz(i));
        zz_merge(idx) = zz_merge(idx) - 1;
        
        
        
        %b. Calculate the proposal probability, q(zz|zz_merge),
        % by computing the Gibbs sampling transition probability from the
        % launch state, zz_launch, to the original split configuration zz.
        % The Gibbs sampling transition probability is th eproduct, over p in
        % S, of the probabilities of setting each zz(p) in the original split
        % state to its original value in a (hypothetical) Gibbs sampling scan
        % from the launch state.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        prod_pp = 0;
        pr=zeros(1,length(S));
        for t=1:length(S)
            p=S(t);
            kk = zz_launch(p); % kk is current component that data item p belongs to
            kk_ = zz(p);
            %if kk==kk_
            %   continue
            %end
            
            
            
            %n(kk) = n(kk) - 1; % subtract from number of data items in component kk
            
            n_launch(zz_launch(p)) = n_launch(zz_launch(p))-1;
            
            v = n_launch(zz_launch(p));
            pr(t)= log(v*(1/(param.ggamma+settings.NN-1)));
            
            lh= compute_likelihood_fast(zz(p), p, R, zz_launch, param, settings);
           
            pr(t) = pr(t)+lh;
%             pr = pr - log(sum(exp(pr)))
%            R prod_pp=prod_pp+pr
%              keyboard
            
            
        end
        prod_pp=log_sum_exp(pr);
        q_zz_merge = prod_pp;
        
        
        
        % evaluate the proposal by the MH acceptance probability
        % alpha(zz_merge, zz)
        % disp('prior_ratio, i j not in the same cluster')
        prior_ratio = -log(param.ggamma) + gammaln(n_merge(zz_merge(i)))  - gammaln(nn(zz(i))) - gammaln(nn(zz(j)));
        
        % likelihood ratio
        % L(zz_merge|R) / L(zz|R) = P(R|zz_merge) / P(R|zz)
        % disp('like_ratio, i j not in the same cluster')
        likelihood_ratio = compute_likelihood_fast(zz_merge(i), i, R, zz_merge, param, settings) - compute_likelihood_fast(zz(i), i, R, zz, param, settings);
        
        % proposal ratio
        % q_ratio_split = q(zz|zz_merge) / q(zz_merge|zz)
        % q(zz|zz_split) = 1
        q_ratio = q_zz_merge - log(1);
        
        zz_new = zz_merge;
        nn_new = n_merge;
        KK_new = KK_merge;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        state = 'merge';
    end
    
    % Metropolis - Hastings
    fr = q_ratio + prior_ratio + likelihood_ratio;
    fr = exp(fr);
    alpha=min(1,fr);
    
    
    if rand(1)<alpha
        % disp('accept new state')
        zz= zz_new;
        nn= nn_new;
        KK= KK_new;
        
    end
    
    %    lkh_vector(iter) = l_v(kk);
    
    %slice sample the hyperparameter ggama
    
    
    if settings.sample_gamma==1
        param.ggamma= slice_sample_gamma(param, zz, nn, KK);
    end
    
    
    %
    %     if param.sample_beta==1
    %     param.bbeta= slice_sample_beta(R, param, zz, nn, KK);
    %     end
    
    g_vector(iter) = param.ggamma;
    b_vector(iter) = param.bbeta;
    
    %combine simple Gibbs sampling for the good of the incremental changes
    [ zz, nn, KK ] = simple_gibbs(zz, nn, R, settings.gibbs_iter, param, settings);
    kk_vector(iter) = KK;
    
    
    zz_cell{iter}=zz;
    lenergy_vector(iter) = compute_energy(zz, nn, KK, settings, param, R);
%    lenergy_vector(iter)
    
    
    %     cluster_freq(iter, :) = compute_most_frequent_clusters(zz,
    %     stat.trueK);
     % save(settings.results_file);
     
     
      %Fill in missing values
            
%     if settings.anyMissing
       [heta] = compute_posterior_weights({zz},  param);
       [param.Rpred_iter{iter}] = heta{1}(zz,zz); 
%         R(~settings.train_mask)=param.Rpred_iter{iter}(~settings.train_mask);
%     end

   if settings.plot_clusters==1
	hold on
	plot_clusters(zz)
   end
end


end


%plot most frequent clusters as a function of the iterations
% for i=1:trueK
%     plot(cluster_freq(:,i));
% end
