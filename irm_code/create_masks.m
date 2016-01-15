function [train_mask test_mask]=create_masks(R, settings)


if settings.masks_given==0
    N=size(R,1);
    
    idxs=find((triu(ones(size(R)))));
    holdout=(settings.hold_out_per)*(length(idxs));
    per=randperm(length(idxs));
    test_mask =sparse(zeros(size(R)));
    t=idxs(per(1:ceil(holdout)));
    test_mask(t)=1;
    % create mask for testing. Make sure masks are symmeric
    test_mask=triu(test_mask).'+triu(test_mask,1);
    train_mask=1-test_mask;
else
    
    train_mask=settings.train_mask;
    test_mask=settings.test_mask;
end


end
