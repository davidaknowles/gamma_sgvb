function cluster_freq = compute_most_frequent_clusters(zz, trueK)
cluster_freq= zeros(1,trueK);

add=0;
for i=1:trueK
    [cluster cluster_freq(i)]= mode(zz);
    ix= find(zz==mode(zz));
    zz(ix)=[];
    cluster_freq = cumsum(cluster_freq)/(sum(cluster_freq));
end


end