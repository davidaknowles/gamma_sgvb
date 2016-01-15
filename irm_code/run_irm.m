% demo of simple IRM model
% "Learning Systems of Concepts with an Infinite Relational Model" by Kemp,
% Tenenbaum and et al.
% Split-merge is implemented for faster mixing

% Konstantina Palla <kp376@cam.ac.uk>
% University of Cambridge
% Created:  January 2012
% Modified: August 2012

% load the settings
[param settings]=load_settings();

settings.random_seed=0;
settings.seed=10;

% initialise parameters (if you don't like the default value provided by the load_settings)
param.ggamma = 1.0 % (CRP concentration parameter)
param.bbeta = 0.1  % (hyperparameter, for the beta prior on the h matrix)




%Create R matrix (NxN) - symmetric
N=settings.NN;
Rtest1 = zeros(N,N);
for i=1:(N/2)
    for j=(1:N/2)
        Rtest1(i,j)= 1;
        Rtest1(j+N/2,i+N/2)=Rtest1(i,j);        
    end
end
Rtest1(10:30,10:30)=0;
param.R=Rtest1;


%% Run inference algorithm
% notice that we learn no weight values (h-values). They are being
% integrated out.


[zz_vector, lenergy_vector, g_vector, b_vector, param, settings] = inference_simple_irm(settings, param, Rtest1);


%% Compute posterior weight matrix
heta_vector = compute_posterior_weights(Rtest1, zz_vector,  param, settings);
   
%% Compute error
err=compute_err( zz_vector,  heta_vector,  Rtest1 , settings , param );

% Compute the Area Under the Curve Estimate
auc = compute_AUC(param, settings, err);
param.auc=auc;
    
%% Save results
filename=strcat('simple_irm_result_', int2str(settings.seed), '.mat');
save(filename, 'param', 'settings', 'err', 'auc');

 






