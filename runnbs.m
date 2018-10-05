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
rng('default');

load(sprintf('%s/groupdata_%s.mat',filepath,listname));

covariates = {
    0   'crsdiag'
    1   'crsr'
    0   'age'
    0   'tbi'
    0   'subjnum'
    0   'days_onset'
};

design_matrix = subjlist.(covariates{1,2});
for c = 2:size(covariates,1)
    design_matrix = cat(2,design_matrix,subjlist.(covariates{c,2}));
end

trange = 0.9:-0.1:0.1;
testdata = squeeze(getfeatures(listname,measure,bandidx,'trange',trange));

%select only patients
testdata = testdata(subjlist.crsdiag < 4,:);
design_matrix = design_matrix(subjlist.crsdiag < 4,:);

fprintf('Estimating GLM with %d permutations and %d covariates on %d subjects...\n', ...
    nperm, size(covariates,1), size(testdata,1));
glm.X = design_matrix;
glm.y = testdata;
glm.contrast = cell2mat(covariates(:,1))';
glm.perms = nperm;
glm.test = 'ttest';
glm.test_stat = NBSglm(glm);

save(sprintf('%s/%s_%s_stats.mat',filepath,listname,bands{bandidx}), 'glm');
fprintf('Done.\n');