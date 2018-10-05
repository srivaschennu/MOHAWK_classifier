function outputs = trainregressor(features,labels,type,varargin)

rng('default');

param = finputcheck(varargin, {
    'mode', 'string', {'cv','holdout','train', 'losocv'}, 'cv'; ...
    'runpca', 'string', {'true','false'}, 'false'; ...
    'covariates', 'float', [], [], ...
    });

holdout = 0.15;
pcaVarExpl = 95/100;
learneropt = {'Standardize',true};
rgrssoropt = {};

switch type
    case {'gpr'}
        hyperparam = {};
end

innercvparam = {'Kfold',4};
numcvfolds = NaN;
if strcmp(param.mode,'cv')
    numcvruns = 25;
    outercv = cvpartition(labels,'Kfold',4);
    numcvfolds = outercv.NumTestSets;
    numfolds = numcvfolds*numcvruns;
    for f = 2:numcvruns
        outercv(f) = repartition(outercv(f-1));
    end
elseif strcmp(param.mode,'losocv')
    numcvruns = 1;
    subjidx = param.covariates;
    param.covariates = [];
    uniqsubj = unique(subjidx);
    numcvfolds = length(uniqsubj);
    numfolds = numcvfolds*numcvruns;
    for s = 1:numcvfolds
        outercv(s).training = true(size(subjidx));
        outercv(s).training(subjidx == uniqsubj(s)) = 0;
    end
elseif strcmp(param.mode,'holdout')
    holdout = 0.15;
    outercv = cvpartition(labels,'HoldOut',holdout);
    numfolds = 25;
    for f = 2:numfolds
        outercv(f) = repartition(outercv(f-1));
    end
elseif strcmp(param.mode,'train')
    outercv = cvpartition(labels,'resubstitution');
    numfolds = 1;
else
    error('Unrecognised mode');
end

clsyfyr.truelabels = labels;
clsyfyr.predlabels = NaN(size(labels,1),numfolds);
clsyfyr.postprob = NaN(size(labels,1),numfolds);

fprintf('Outer fold ');
for f = 1:numfolds
    if f > 1
        fprintf(repmat('\b',1,length(progstr)));
    end
    progstr = sprintf('%d/%d',f,numfolds);
    fprintf('%s',progstr);
    
    if strcmp(param.mode,'cv')
        trainidx = training(outercv(floor((f-1)/numcvfolds)+1),mod((f-1),numcvfolds)+1);
        testidx = test(outercv(floor((f-1)/numcvfolds)+1),mod((f-1),numcvfolds)+1);
    elseif strcmp(param.mode,'losocv')
        trainidx = outercv(f).training;
        testidx = ~trainidx;
    elseif strcmp(param.mode,'train')
        trainidx = training(outercv(f));
        testidx = ~trainidx;
    else
        trainidx = training(outercv(f));
        testidx = test(outercv(f));
    end
    
    trainfeat = features(trainidx,:);
    testfeat = features(testidx,:);
    trainlabels = labels(trainidx);
    testlabels = labels(testidx);
    
    if size(trainfeat,2) > 1 && strcmp(param.runpca,'true')
        [pcaCoeff, ~, ~, ~, explained] = pca(trainfeat,'Centered',true);
        numPCAComponentsToKeep = find(cumsum(explained)/sum(explained) >= pcaVarExpl, 1);
        clsyfyr.pcaCoeff = pcaCoeff(:,1:numPCAComponentsToKeep);
        trainfeat = trainfeat * clsyfyr.pcaCoeff;
        testfeat = testfeat * clsyfyr.pcaCoeff;
    else
        clsyfyr.pcaCoeff = [];
    end
        
    if ~isempty(param.covariates)
        trainfeat = cat(2,param.covariates(trainidx,:),trainfeat);
        testfeat = cat(2,param.covariates(testidx,:),testfeat);
    end
    
    switch type
        case {'gpr'}
            clsyfyr.perf(f) = getperf(fitrgp(trainfeat, trainlabels, innercvparam{:}, 'OptimizeHyperparameters','auto'));
    end
end
fprintf('\nDone.\n');

if strcmp(param.mode,'cv') || strcmp(param.mode,'losocv')
    for f = 1:numfolds/numcvfolds
        cm = confusionmat(labels, nansum(clsyfyr.predlabels(:,(f-1)*numcvfolds+1:f*numcvfolds),2), 'order', unique(labels));
        clsyfyr.testcm(:,:,f) = cm;
        clsyfyr.testperf(f) = cm2perf(cm);
    end
end

clsyfyr.funcopt = param;
clsyfyr.learneropt = learneropt;
clsyfyr.cvopt = innercvparam;
clsyfyr.clsyfyropt = clsyfyropt;
clsyfyr.numfolds = numfolds;
clsyfyr.numcvfolds = numcvfolds;

if strcmp(param.mode,'train')
    outputs = {clsyfyr model};
else
    outputs = {clsyfyr};
end

end

function perf = gridsearch(features,labels,type,clsyfyropt,cvopt,learneropt,hyperparam)

switch type
    case 'knn'
        Nvals = hyperparam{1};
        perf = zeros(size(Nvals));
        for n = 1:length(Nvals)
            model = fitcecoc(features,labels,clsyfyropt{:},cvopt{:}, ...
                'Learners',templateKNN(learneropt{:},'NumNeighbors',Nvals(n)));
            perf(n) = getperf(model,labels);
        end
        
    case 'svm-linear'
        Cvals = hyperparam{1};
        perf = zeros(size(Cvals));
        for c = 1:length(Cvals)
            model = fitcecoc(features,labels,clsyfyropt{:},cvopt{:}, ...
                'Learners',templateSVM(learneropt{:},'BoxConstraint',Cvals(c)));
            perf(c) = getperf(model,labels);
        end
        
    case 'svm-rbf'
        Cvals = hyperparam{1};
        Kvals = hyperparam{2};
        perf = zeros(length(Cvals),length(Kvals));
        for c = 1:length(Cvals)
            for k = 1:length(Kvals)
                model = fitcecoc(features,labels,clsyfyropt{:},cvopt{:}, ...
                    'Learners',templateSVM(learneropt{:},'KernelFunction','RBF',...
                    'BoxConstraint',Cvals(c),'KernelScale',Kvals(k)));
                perf(c,k) = getperf(model,labels);
            end
        end
        
    case 'tree'
        Lvals = hyperparam{1};
        perf = zeros(size(Lvals));
        for l = 1:length(Lvals)
%             model = fitcecoc(features,labels,clsyfyropt{:},cvopt{:}, ...
%                 'Learners',templateTree('MinLeafSize',Lvals(l)));

            model = fitensemble(features,labels,'AdaBoostM1',100, ...
                templateTree('MaxNumSplits',Lvals(l)),cvopt{:},'LearnRate',0.1,'Prior','uniform');
            perf(l) = getperf(model,labels);
        end
end
end

function perf = getperf(model,labels)
perf = kfoldLoss(model);
end
