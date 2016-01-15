
dataset='NIPS234'; 

addpath EPM
addpath examples
addpath irm_code/
addpath irm_code/utils

TrainRatio =.8;

IsDisplay = false;

Burnin=1500;
Collections=1500;

gsgvbIndex=1; 
mcmcIndex=2; 
irmIndex=3;
mapIndex=4; 
nsgvbIndex=5; 
hmcIndex=6; 

assessPerformance=@(pred,truth) [ aucROC(pred,truth) ...
    mean(abs(pred-truth)) mean(truth.*log(pred)+(1-truth).*log(1-pred)) ]; 

Ks=[10 20 30 50 70 100]; 

allresults={}; 

for ki=1:length(Ks)
    
    parfor state=0:9

        load nips_small
        B=triu(R,1); 
        N=size(B,1);
        
        K=Ks(ki);

        rng(state,'twister');
        [idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);

        missing=~BTrain_Mask;

        settings=gammaSGVBsettings(6);
        settings.samples=1000; 
        settings.plot=0;
        settings.rho=.95; 
        settings.forgetting=.1; 
        settings.testGrad=0;
        tic

         try 
            [~,ll,Xpred]=epmMAP(B+B',K,missing,settings); 
            results{state+1}(mapIndex,:)=[ assessPerformance(Xpred(idx_test),B(idx_test))...
            assessPerformance(Xpred(idx_train),B(idx_train)) toc ];
         catch
             results{state+1}(mapIndex,:)=NaN*zeros(1,7);
         end

        try 
            tic
            [~,~,~,Xpred]=epmNormalSGVB(B+B',K,missing,settings); 
            results{state+1}(nsgvbIndex,:)=[ assessPerformance(Xpred(idx_test),B(idx_test))...
                assessPerformance(Xpred(idx_train),B(idx_train)) toc ];
         catch
             results{state+1}(nsgvbIndex,:)=NaN*zeros(1,7);
         end

        tic
        [~,~,~,Xpred]=epmSGVBnoMu(B+B',K,missing,settings); 
        results{state+1}(gsgvbIndex,:)=[ assessPerformance(Xpred(idx_test),B(idx_test))...
            assessPerformance(Xpred(idx_train),B(idx_train)) toc ];
        
        [param, settings]=load_settings();
        settings.masks_given=1; 
        settings.numiter=100; % the IRM code does split-merge moves so 100 iterations is not actually that low
        settings.train_mask=~missing;
        settings.test_mask=missing;
        param.R=B+B'; 
        tic
        [zz_vector, lenergy_vector, g_vector, b_vector, param, settings] = inference_simple_irm(settings, param, param.R);

        Rpred=param.R*0; 
        count=0; 
        for i=51:100 % average over the last 50 iterations
            Rpred=Rpred+param.Rpred_iter{i}; 
            count=count+1; 
        end
        ProbAve=Rpred/count;
        results{state+1}(irmIndex,:)=[ assessPerformance(ProbAve(idx_test),B(idx_test))...
            assessPerformance(ProbAve(idx_train),B(idx_train)) toc ];  
   
        %Datatype='Count';
        Datatype='Binary';
        Modeltype = 'Infinite';
        %Modeltype = 'Finite';

        tic
        [AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z] = GP_EPM(B,K,idx_train,idx_test,Burnin,Collections, IsDisplay, Datatype, Modeltype);
        %fprintf('GP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);
        results{state+1}(mcmcIndex,:)=[ assessPerformance(ProbAve(idx_test),B(idx_test))...
            assessPerformance(ProbAve(idx_train),B(idx_train)) toc ];

    end

    allresults{ki}=results;

end

% 1-3: test set AUC, accuracy, log likelihood
% 4-6: train AUC, accuracy, log likelihood
% 7: time 
to_plot=1; 

results_flat=zeros(5,7,length(Ks),10); 
for k=1:length(Ks)
    for i=1:10
        results_flat(:,:,k,i)=allresults{k}{i}; 
    end
end

mean_across_runs=mean(results_flat,4); 
std_across_runs=std(results_flat,0,4); 

test_set_auc_means=squeeze(mean_across_runs(:,to_plot,:)); 
test_set_auc_std=squeeze(std_across_runs(:,to_plot,:));
clf
hold off
errorbar(Ks,test_set_auc_means(1,:),test_set_auc_std(1,:))
hold on
for i=2:5
    errorbar(Ks+(i-1)*.3,test_set_auc_means(i,:),test_set_auc_std(i,:))
end
legend({'GammaSGVB' 'MCMC' 'IRM' 'MAP' 'NormSGVB'})
ylabel('test set AUC'); 
xlabel('truncation level');
set(gca,'fontsize',14);
set(gcf,'color','white');
xlim([0 110])
lh=findall(gcf,'tag','legend');
set(lh,'location','eastoutside');
