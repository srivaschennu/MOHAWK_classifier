function outputs = trainclassifier(features,labels,type,varargin)

rng('default');

param = finputcheck(varargin, {
    'oversample', 'string', {'true','false'}, 'true'; ...
    'mode', 'string', {'cv','holdout','train', 'losocv'}, 'cv'; ...
    'runpca', 'string', {'true','false'}, 'false'; ...
    'prior', 'string', {'empirical','uniform'}, 'uniform'; ...
    'posterior', 'string', {'on','off'}, 'off'; ...    
    'covariates', 'float', [], [], ...
    });

holdout = 0.15;
pcaVarExpl = 95/100;
learneropt = {'Standardize',true};
clsyfyropt = {'Prior',param.prior,'FitPosterior',param.posterior};

switch type
    case 'knn'
        Nvals = 1:10;
        hyperparam = {Nvals};
        
    case 'svm-linear'
        Cvals = [.001 .01 .1 .2 .5 1 2 10];
        hyperparam = {Cvals};
        
    case 'svm-rbf'
        Cvals = [.001 .01 .1 .2 .5 1 2 10];
        Kvals = [.001 .01 .1 .2 .5 1 2 10];
        hyperparam = {Cvals, Kvals};
        
    case 'tree'
        Lvals = 2.^(0:floor(log2(size(features,1)-1)));
        hyperparam = {Lvals};
        
    case {'nbayes' 'nn'}
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
    
    if strcmp(param.oversample,'true')
        classes = unique(trainlabels);
        classcounts = zeros(1,length(classes));
        for c = 1:length(classes)
            classcounts(c) = sum(trainlabels == classes(c));
        end
        [~,sortidx] = sort(classcounts);
        classes = classes(sortidx);
        
        for c = 1:length(classes)-1
            newfeat = ADASYN(trainfeat(trainlabels == classes(c) | trainlabels == classes(end),:),...
                trainlabels(trainlabels == classes(c) | trainlabels == classes(end)) == classes(c),...
                [],[],[],false);
            trainfeat = cat(1,trainfeat,newfeat);
            trainlabels = cat(1,trainlabels,ones(size(newfeat,1),1)*classes(c));
        end
    end
    
    if ~isempty(param.covariates) && strcmp(param.oversample,'false')
        trainfeat = cat(2,param.covariates(trainidx,:),trainfeat);
        testfeat = cat(2,param.covariates(testidx,:),testfeat);
    elseif ~isempty(param.covariates) && strcmp(param.oversample,'true')
        warning('Covariates ignored due to feature oversampling!');
    end
    
    switch type
        case {'knn' 'svm-linear' 'svm-rbf' 'tree' 'nbayes'}
            if ~isempty(hyperparam)
                perf = gridsearch(trainfeat, trainlabels, type, clsyfyropt, innercvparam, learneropt, hyperparam);
                [~,maxidx] = max(perf(:));
            end
            
            switch type
                case 'knn'
                    bestN = ind2sub(size(perf),maxidx);
                    learners = templateKNN(learneropt{:},'NumNeighbors', Nvals(bestN));
                    
                case 'svm-linear'
                    bestC = ind2sub(size(perf),maxidx);
                    learners = templateSVM(learneropt{:},'BoxConstraint',Cvals(bestC));
                    
                case 'svm-rbf'
                    [bestC,bestK] = ind2sub(size(perf),maxidx);
                    learners = templateSVM(learneropt{:},'KernelFunction','RBF','BoxConstraint',Cvals(bestC),'KernelScale',Kvals(bestK));
                    
                case 'tree'
                    bestL = ind2sub(size(perf),maxidx);
                    learners = templateTree('MaxNumSplits', Lvals(bestL));
                    
                case 'nbayes'
                    learners = templateNaiveBayes('DistributionNames', 'kernel');
                    
                otherwise
                    error('Classifier type not recognised');
            end
            
            if strcmp(type,'tree')
                [clsyfyr.perf(f),clsyfyr.cm(:,:,f),bestthresh] = getperf(fitensemble(trainfeat, trainlabels, 'AdaBoostM1', 100, ...
                    learners, innercvparam{:}, 'LearnRate', 0.1, 'Prior', 'uniform'), trainlabels);
                model = fitensemble(trainfeat, trainlabels, 'AdaBoostM1', 100, learners, 'LearnRate', 0.1, 'Prior', 'uniform');
            else
                [clsyfyr.perf(f),clsyfyr.cm(:,:,f),bestthresh] = getperf(fitcecoc(trainfeat, trainlabels, clsyfyropt{:}, innercvparam{:}, ...
                    'Learners', learners), trainlabels);
                model = fitcecoc(trainfeat, trainlabels, clsyfyropt{:}, 'Learners', learners);
            end
            
            if strcmp(model.ScoreType,'junk')
                [~,~,~,postprob] = predict(model, testfeat);
                clsyfyr.predlabels(testidx,f) = double(postprob(:,end) > bestthresh);
            else
                clsyfyr.predlabels(testidx,f) = predict(model, testfeat);
            end
            
%         case 'nn'
%             innercvparam = cvpartition(trainlabels,'HoldOut',holdout);
%             itrainfeat = trainfeat(training(innercvparam),:);
%             itrainlabels = trainlabels(training(innercvparam));
%             ivalfeat = trainfeat(test(innercvparam),:);
%             ivallabels = trainlabels(test(innercvparam));
%             
%             thisfeat = cat(1,itrainfeat,ivalfeat,testfeat);
%             thislab = cat(1,itrainlabels,ivallabels,testlabels);
%             
%             hiddenLayerSize = size(thisfeat,2)*2;
%             model = patternnet(hiddenLayerSize);
%             model.trainParam.showWindow = 0;
%             
%             model.divideFcn = 'divideind';
%             model.divideParam.trainInd = 1:length(itrainlabels);
%             model.divideParam.valInd = length(itrainlabels)+1:length(itrainlabels)+length(ivallabels);
%             model.divideParam.testInd = length(itrainlabels)+length(ivallabels)+1:...
%                 length(itrainlabels)+length(ivallabels)+length(testlabels);
%             
%             inputs = thisfeat';
%             targets = full(ind2vec(thislab'+1));
%             model = train(model,inputs,targets);
%             outputs = model(inputs);
%             
%             [~,cm] = confusion(targets(:,model.divideParam.valInd),...
%                 outputs(:,model.divideParam.valInd));
%             clsyfyr.cm(:,:,f) = cm;
%             clsyfyr.perf(f) = cm2perf(cm);
%             
%             clsyfyr.predlabels(testidx,f) = ...
%                 vec2ind(compet(outputs(:,model.divideParam.testInd)))-1;
    end
    if strcmp(param.mode,'holdout')
        cm = confusionmat(testlabels, clsyfyr.predlabels(testidx,f), 'order', unique(testlabels));
        clsyfyr.testcm(:,:,f) = cm;
        clsyfyr.testperf(f) = cm2perf(cm);
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

function [perf,cm,bestthresh] = getperf(model,labels)

if strcmp(model.ScoreType,'junk')
    [~,~,~,postprob] = kfoldPredict(model);

    predlabels = nan(size(postprob,1),1);
    for f = 1:model.Partition.NumTestSets
        trainidx = training(model.Partition,f);
        testidx = test(model.Partition,f);
        [x,y,t] = perfcurve(labels(trainidx),postprob(trainidx,end),max(labels(trainidx)));
        [~,bestthresh] = max(abs(y + (1-x) - 1));
        bestthresh = t(bestthresh);
        predlabels(testidx) = double(postprob(testidx,end) > bestthresh);
    end
    
    [x,y,t] = perfcurve(labels,postprob(:,end),max(labels));
    [~,bestthresh] = max(abs(y + (1-x) - 1));
    bestthresh = t(bestthresh);    
else
    predlabels = kfoldPredict(model);
    bestthresh = NaN;
end

cm = confusionmat(labels,predlabels,'order',unique(labels));
perf = cm2perf(cm);
end

function perf = cm2perf(cm)
% perf = mean( diag( cm ./ repmat(sum(cm,2),1,size(cm,2)) ) );

for i = 1:size(cm,1)
    precision(i) = cm(i,i)/(sum(cm(:,i))+eps);
    recall(i) = cm(i,i)/(sum(cm(i,:))+eps);
end

perf = 2 * ( (precision .* recall) ./ (precision + recall + eps) );
perf = mean(perf);
end