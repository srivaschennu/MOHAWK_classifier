group = 'allpat';
groups = [0 1];
groupnames = {'UWS','MCS-'};
type = 'svm-rbf';
mode = 'train';
suffix = sprintf('%s_%s_%s_%s_%s',group,type,groupnames{1},groupnames{2},mode);

[clust_job,clsyfyrinfo] = runclassifierjob(group,'phoenix','trainclassifier',{type,'runpca','false','mode',mode},'groups',groups,'groupnames',groupnames);
saveclassifier(clust_job,clsyfyrinfo,suffix);

[clust_job,clsyfyrinfo] = runclassifierjob('allpat','phoenix','trainclassifier',{'svm-rbf','runpca','false','mode','train'});
[clust_job2,clsyfyrinfo2] = runclassifierjob('allpat','phoenix','trainclassifier',{'tree','runpca','true','mode','train'});
[clust_job3,clsyfyrinfo3] = runclassifierjob('allpat','phoenix','trainclassifier',{'nn','runpca','true','mode','train'});
[clust_job4,clsyfyrinfo4] = runclassifierjob('allpat','phoenix','trainclassifier',{'knn','runpca','true','mode','train'});
[clust_job5,clsyfyrinfo5] = runclassifierjob('allpat','phoenix','trainclassifier',{'nbayes','runpca','true','mode','train'});


saveclassifier(clust_job,clsyfyrinfo,'allpat_svm-rbf_UWS_MCS-_train');
saveclassifier(clust_job2,clsyfyrinfo2,'allpat_tree_UWS_MCS-_train');
saveclassifier(clust_job3,clsyfyrinfo3,'allpat_nn_UWS_MCS-_train');
saveclassifier(clust_job4,clsyfyrinfo4,'allpat_knn_UWS_MCS-_train');
saveclassifier(clust_job5,clsyfyrinfo5,'allpat_nbayes_UWS_MCS-_train');

%combclassifier({'allpat_svm-rbf_UWS_MCS-_train','allpat_tree_UWS_MCS-_train','allpat_nn_UWS_MCS-_train','allpat_knn_UWS_MCS-_train','allpat_nbayes_UWS_MCS-_train'});
