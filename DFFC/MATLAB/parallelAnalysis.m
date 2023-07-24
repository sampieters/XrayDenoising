function [V1, D1,numberPC]=parallelAnalysis(flatFields,repetitions)
% selection of the number of components for PCA using parallel Analysis.
% Each flat field is a single row of the matrix flatFields, different
% rows are different observations.

stdEFF=std(flatFields,0,2);
keepTrack=zeros(size(flatFields,2),repetitions);
stdMatrix=repmat(stdEFF,[1,size(flatFields,2)]);

rng('default');
lol = randn(size(flatFields));
save("random.mat", "lol")
for ii=1:repetitions
    display(['Parallel Analysis: repetition ' int2str(ii)]);
    sample=stdMatrix .* lol;
    hu = cov(sample);
    [~, D1] = eig(cov(sample));
    D1=diag(D1);
    keepTrack(:,ii)=D1;
end

mean_flat_fields_EFF=mean(flatFields,2);
F=flatFields-repmat(mean_flat_fields_EFF,[1,size(flatFields,2)]);
cov2 = cov(F);
[V1, D1] = eig(cov(F));
D1=diag(D1);

selection=zeros(1,size(flatFields,2));

mean_keepTrack = mean(keepTrack,2);
std_keepTrack = std(keepTrack,0,2);
f = (mean(keepTrack,2)+2*std(keepTrack,0,2));

selection(D1>(mean(keepTrack,2)+2*std(keepTrack,0,2)))=1;
numberPC=sum(selection);
end