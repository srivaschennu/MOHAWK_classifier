function runnbs(listname,measure,bandidx,varargin)

loadpaths

bands = {
    'delta'
    'theta'
    'alpha'
    'beta'
    'gamma'
    };

nperm = 2000;
siglevel = 0.05;
rng('default');

load(sprintf('%s/groupdata_%s.mat',filepath,listname));

covariates = {
    'age'
    'tbi'
    'crsdiag'
    'crsr'
};

design_matrix = subjlist.(covariates{1});
for c = 2:length(covariates)
    design_matrix = cat(2,design_matrix,subjlist.(covariates{c}));
end

trange = 0.9:-0.1:0.1;
testdata = squeeze(getfeatures(listname,measure,bandidx,'trange',trange));

%select only patients
testdata = testdata(subjlist.crsdiag < 5,:);
design_matrix = design_matrix(subjlist.crsdiag < 5,:);

fprintf('Estimating GLM...\n');
glm.X = design_matrix;
glm.y = testdata;
glm.contrast = [0 0 0 1];
glm.perms = nperm;
glm.test = 'ttest';
test_stat = NBSglm(glm);

fprintf('Running permutation statistics...\n');
stats.alpha = 0.05;
stats.thresh = quantile(test_stat(1,:), 1-siglevel);
stats.size = 'extent';
stats.N = size(allcoh,3);
stats.test_stat = test_stat;

[~,n_nets,netmask,netpval] = evalc('NBSstats(stats)');

save(sprintf('%s/%s_%s_stats.mat',filepath,listname,bands{bandidx}),...
    'test_stat','n_nets','netmask','netpval');
fprintf('Done.\n');

