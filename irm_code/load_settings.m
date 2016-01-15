function [param settings]= load_settings()

% number of true clusters
settings.trueK = 5;
% number of data points
settings.NN = 40;
% number of Gibbs iterations
settings.numiter = 100;
% number of intermediate restricted Gibbs sampling scans (for split merge)
settings.numiter_rstr_gibbs =5;
settings.gibbs_iter = 1;
% decide whether to sample the CRP hyperparameter \gamma
settings.sample_gamma=1;

% define whether the test and train mask is given or is randomly created
settings.masks_given=0;
% percentage of the data links kept as test data for the randomly created masks.
settings.hold_out_per=0.2;

% the number of iterations that will be used for the computation of the
% errror. This is the number of samples taken into account from the end of
% the sampling process. (the ones proceeded will be considered as burn in
% period)
settings.evaluation_window = 0.3*settings.numiter;


% plot clusters or not over the iterations
settings.plot_clusters=0;



%initialize the seed. Use a fixed seed when you want to compare the
%algorithm over different modifications.
settings.random_seed=1;
settings.seed=10;

%default values of the gamma and beta parameters
param.ggamma = 1.0; % (CRP concentration parameter)
param.bbeta = 0.1;  % (hyperparameter, for the beta prior on the h matrix)


end
