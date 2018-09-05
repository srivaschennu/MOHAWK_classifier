group = 'allpat';
groups = [0 1];
groupnames = {'UWS','MCS-'};
type = 'svm-rbf';
mode = 'train';
suffix = sprintf('%s_%s_%s_%s_%s',group,type,groupnames{1},groupnames{2},mode);

[clust_job,clsyfyrinfo] = runclassifierjob(group,'phoenix','trainclassifier',{type,'runpca','false','mode',mode},'groups',groups,'groupnames',groupnames);
saveclassifier(clust_job,clsyfyrinfo,suffix);

[clust_job,clsyfyrinfo] = runclassifierjob('betadoc','phoenix','trainclassifier',{'svm-rbf','runpca','false','mode','losocv'},'group','crsdiag','covariates','subjnum');
[clust_job2,clsyfyrinfo2] = runclassifierjob('betadoc','phoenix','trainclassifier',{'tree','runpca','true','mode','losocv'},'group','crsdiag','covariates','subjnum');
[clust_job3,clsyfyrinfo3] = runclassifierjob('betadoc','phoenix','trainclassifier',{'knn','runpca','true','mode','losocv'},'group','crsdiag','covariates','subjnum');
[clust_job4,clsyfyrinfo4] = runclassifierjob('betadoc','phoenix','trainclassifier',{'nbayes','runpca','true','mode','losocv'},'group','crsdiag','covariates','subjnum');
% [clust_job3,clsyfyrinfo3] = runclassifierjob('allpat','phoenix','trainclassifier',{'nn','runpca','true','mode','train'});

saveclassifier(clust_job,clsyfyrinfo,'betadoc_svm-rbf_UWS_MCS-_losocv');
saveclassifier(clust_job2,clsyfyrinfo2,'betadoc_tree_UWS_MCS-_losocv');
saveclassifier(clust_job3,clsyfyrinfo3,'betadoc_knn_UWS_MCS-_losocv');
saveclassifier(clust_job4,clsyfyrinfo4,'betadoc_nbayes_UWS_MCS-_losocv');
% saveclassifier(clust_job3,clsyfyrinfo3,'allpat_nn_UWS_MCS-_train');

%combclassifier({'allpat_svm-rbf_UWS_MCS-_train','allpat_tree_UWS_MCS-_train','allpat_nn_UWS_MCS-_train','allpat_knn_UWS_MCS-_train','allpat_nbayes_UWS_MCS-_train'});
