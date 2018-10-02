group = 'allpat';
groups = [0 1];
groupnames = {'UWS','MCS-'};
type = 'svm-rbf';
mode = 'train';
suffix = sprintf('%s_%s_%s_%s_%s',group,type,groupnames{1},groupnames{2},mode);

[clust_job,clsyfyrinfo] = runclassifierjob(group,'phoenix','trainclassifier',{type,'runpca','false','mode',mode},'groups',groups,'groupnames',groupnames);
saveclassifier(clust_job,clsyfyrinfo,suffix);

[clust_job,clsyfyrinfo] = runclassifierjob('betadoc','phoenix','trainclassifier',{'svm-linear','runpca','false','mode','cv'},'group','crsdiag');
[clust_job3,clsyfyrinfo3] = runclassifierjob('betadoc','phoenix','trainclassifier',{'tree','runpca','true','mode','cv'},'group','crsdiag');
[clust_job4,clsyfyrinfo4] = runclassifierjob('betadoc','phoenix','trainclassifier',{'knn','runpca','true','mode','cv'},'group','crsdiag');
[clust_job5,clsyfyrinfo5] = runclassifierjob('betadoc','phoenix','trainclassifier',{'nbayes','runpca','true','mode','cv'},'group','crsdiag');
[clust_job2,clsyfyrinfo2] = runclassifierjob('betadoc','phoenix','trainclassifier',{'svm-rbf','runpca','false','mode','cv'},'group','crsdiag');

[clust_job,clsyfyrinfo] = runclassifierjob('betadoc','phoenix','trainclassifier',{'tree','runpca','false','mode','cv'},'group','crsdiag');
saveclassifier(clust_job,clsyfyrinfo,'betadoc_tree_UWS_MCS-_cv');

[~,clsyfyrinfo] = runclassifierjob('allnewsubj','local','trainclassifier',{'tree','runpca','false','mode','cv'});
saveclassifier(clust_job,clsyfyrinfo,'allnewsubj_svm-rbf_UWS_MCS-_cv');

saveclassifier(clust_job,clsyfyrinfo,'betadoc_svm-linear_UWS_MCS-_cv','group','crsdiag');
saveclassifier(clust_job3,clsyfyrinfo3,'betadoc_tree_UWS_MCS-_cv','group','crsdiag');
saveclassifier(clust_job4,clsyfyrinfo4,'betadoc_knn_UWS_MCS-_cv','group','crsdiag');
saveclassifier(clust_job5,clsyfyrinfo5,'betadoc_nbayes_UWS_MCS-_cv','group','crsdiag');
saveclassifier(clust_job2,clsyfyrinfo2,'betadoc_svm-rbf_UWS_MCS-_cv','group','crsdiag');

% saveclassifier(clust_job3,clsyfyrinfo3,'allpat_nn_UWS_MCS-_train');

%combclassifier({'allpat_svm-rbf_UWS_MCS-_train','allpat_tree_UWS_MCS-_train','allpat_nn_UWS_MCS-_train','allpat_knn_UWS_MCS-_train','allpat_nbayes_UWS_MCS-_train'});
