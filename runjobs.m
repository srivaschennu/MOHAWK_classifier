group = 'allsubj';
groups = [0 1];
groupnames = {'UWS','MCS-'};
type = 'svm-rbf';
mode = 'train';
suffix = sprintf('%s_%s_%s_%s_%s',group,type,groupnames{1},groupnames{2},mode);

[clust_job,clsyfyrinfo] = runclassifierjob(group,'phoenix','trainclassifier',{type,'runpca','false','mode',mode},'groups',groups,'groupnames',groupnames);
saveclassifier(clust_job,clsyfyrinfo,suffix);

[clust_job,clsyfyrinfo] = runclassifierjob('allsubj','phoenix','trainclassifier',{'svm-rbf','runpca','false','mode','cv'});
[clust_job2,clsyfyrinfo2] = runclassifierjob('allsubj','phoenix','trainclassifier',{'tree','runpca','true','mode','cv'});
[clust_job3,clsyfyrinfo3] = runclassifierjob('allsubj','phoenix','trainclassifier',{'nn','runpca','true','mode','cv'});
[clust_job4,clsyfyrinfo4] = runclassifierjob('allsubj','phoenix','trainclassifier',{'knn','runpca','true','mode','cv'});
[clust_job5,clsyfyrinfo5] = runclassifierjob('allsubj','phoenix','trainclassifier',{'nbayes','runpca','true','mode','cv'});


saveclassifier(clust_job,clsyfyrinfo,'allsubj_svm-rbf_UWS_MCS-_cv');
saveclassifier(clust_job2,clsyfyrinfo2,'allsubj_tree_UWS_MCS-_cv');
saveclassifier(clust_job3,clsyfyrinfo3,'allsubj_nn_UWS_MCS-_cv');
saveclassifier(clust_job4,clsyfyrinfo4,'allsubj_knn_UWS_MCS-_cv');
saveclassifier(clust_job5,clsyfyrinfo5,'allsubj_nbayes_UWS_MCS-_cv');

combclassifier({'allsubj_svm-rbf_UWS_MCS-_cv','allsubj_tree_UWS_MCS-_cv','allsubj_nn_UWS_MCS-_cv','allsubj_knn_UWS_MCS-_cv','allsubj_nbayes_UWS_MCS-_cv'});
