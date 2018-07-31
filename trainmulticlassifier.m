function outputs = trainmulticlassifier(features,labelset,type,varargin)

rng('default');

param = finputcheck(varargin, {
    'oversample', 'string', {'true','false'}, 'true'; ...
    'mode', 'string', {'cv','holdout','train'}, 'cv'; ...
    'runpca', 'string', {'true','false'}, 'false'; ...
    'prior', 'string', {'empirical','uniform'}, 'uniform'; ...
    'covariates', 'float', [], [], ...
    });

holdout = 0.15;
pcaVarExpl = 95/100;
clsyfyropt = {'Standardize',true};
innercvparam = {'Prior',param.prior,'KFold',4};

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
        Lvals = 1:15;
        hyperparam = {Lvals};
        
    case {'nbayes' 'nn'}
        hyperparam = {};
end

codinglabels = [0 1 2];
codingscheme = [
    0   0
    1   0
  NaN   1
    ];

for s = 1:size(codingscheme,2)
    labels = labelset;
    for i = 1:size(codingscheme,1)
        labels(labels == codinglabels(i)) = codingscheme(i,s);
    end
    nanidx = isnan(labels);
    nanfeat = features(nanidx,:);
    
    numcvfolds = NaN;
    if strcmp(param.mode,'cv')
        numcvruns = 25;
        outercv = cvpartition(labels,'Kfold',4);
        numcvfolds = outercv.NumTestSets;
        numfolds = numcvfolds*numcvruns;
        for f = 2:numcvruns
            outercv(f) = repartition(outercv(f-1));
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
    
    clsyfyr(s).truelabels = labels;
    clsyfyr(s).predlabels = NaN(size(labels,1),numfolds);
    
    fprintf('Outer fold');
    for f = 1:numfolds
        fprintf(' %d',f);
        
        if strcmp(param.mode,'cv')
            trainidx = training(outercv(floor((f-1)/numcvfolds)+1),mod((f-1),numcvfolds)+1);
            testidx = test(outercv(floor((f-1)/numcvfolds)+1),mod((f-1),numcvfolds)+1);
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
            clsyfyr(s).pcaCoeff = pcaCoeff(:,1:numPCAComponentsToKeep);
            trainfeat = applypca(trainfeat,clsyfyr(s).pcaCoeff);
            testfeat = applypca(testfeat,clsyfyr(s).pcaCoeff);
        else
            clsyfyr(s).pcaCoeff = [];
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
                    perf = gridsearch(trainfeat, trainlabels, type, innercvparam, clsyfyropt, hyperparam);
                    [~,maxidx] = max(perf(:));
                end
                
                switch type
                    case 'knn'
                        bestN = ind2sub(size(perf),maxidx);
                        learners = templateKNN(clsyfyropt{:},'NumNeighbors', Nvals(bestN));
                        
                    case 'svm-linear'
                        [bestC] = ind2sub(size(perf),maxidx);
                        learners = templateSVM(clsyfyropt{:},'BoxConstraint',Cvals(bestC));
                        
                    case 'svm-rbf'
                        [bestC,bestK] = ind2sub(size(perf),maxidx);
                        learners = templateSVM(clsyfyropt{:},'KernelFunction','RBF','BoxConstraint',Cvals(bestC),'KernelScale',Kvals(bestK));
                        
                    case 'tree'
                        bestL = ind2sub(size(perf),maxidx);
                        learners = templateTree('MinLeafSize', Lvals(bestL));
                        
                    case 'nbayes'
                        learners = templateNaiveBayes('DistributionNames', 'kernel');
                end
                [clsyfyr(s).perf(f),clsyfyr(s).cm(:,:,f)] = getperf(fitcecoc(trainfeat, trainlabels, innercvparam{:}, ...
                    'Learners', learners), trainlabels);
                
                model = fitcecoc(trainfeat, trainlabels, 'Prior', param.prior, 'Learners', learners);
                clsyfyr(s).predlabels(testidx,f) = predict(model, testfeat);
                clsyfyr(s).predlabels(nanidx,f) = predict(model, applypca(nanfeat,clsyfyr(s).pcaCoeff));
                
            case 'nn'
                innercvparam = cvpartition(trainlabels,'HoldOut',holdout);
                itrainfeat = trainfeat(training(innercvparam),:);
                itrainlabels = trainlabels(training(innercvparam));
                ivalfeat = trainfeat(test(innercvparam),:);
                ivallabels = trainlabels(test(innercvparam));
                
                thisfeat = cat(1,itrainfeat,ivalfeat);
                thislab = cat(1,itrainlabels,ivallabels);
                
                hiddenLayerSize = size(thisfeat,2)*2;
                model = patternnet(hiddenLayerSize);
                model.trainParam.showWindow = 0;
                
                model.divideFcn = 'divideind';
                model.divideParam.trainInd = 1:length(itrainlabels);
                model.divideParam.valInd = length(itrainlabels)+1:length(itrainlabels)+length(ivallabels);
                
                inputs = thisfeat';
                targets = full(ind2vec(thislab'+1));
                model = train(model,inputs,targets);
                outputs = model(inputs);
                
                [~,cm] = confusion(targets(:,model.divideParam.valInd),...
                    outputs(:,model.divideParam.valInd));
                clsyfyr(s).cm(:,:,f) = cm;
                normcm = cm ./ repmat(sum(cm,2),1,size(cm,2));
                clsyfyr(s).perf(f) = mean(diag(normcm));
                
                clsyfyr(s).predlabels(testidx,f) = ...
                    vec2ind(compet(model(testfeat')))-1;
                
                clsyfyr(s).predlabels(nanidx,f) = ...
                    vec2ind(compet(model(applypca(nanfeat,clsyfyr(s).pcaCoeff)')))-1;
        end
        if strcmp(param.mode,'holdout')
            cm = confusionmat(testlabels, clsyfyr(s).predlabels(testidx,f), 'order', unique(testlabels));
            clsyfyr(s).testcm(:,:,f) = cm;
            clsyfyr(s).testperf(f) = mean( diag( cm ./ repmat(sum(cm,2),1,size(cm,2)) ) );
        end
    end
    fprintf(' Done.\n');
    
    if strcmp(param.mode,'cv')
        for f = 1:numfolds/numcvfolds
            thisfold = (f-1)*numcvfolds+1:f*numcvfolds;
            cm = confusionmat(labels(~nanidx), nansum(clsyfyr(s).predlabels(~nanidx,thisfold),2));
            clsyfyr(s).testcm(:,:,f) = cm;
            clsyfyr(s).testperf(f) = mean( diag( cm ./ repmat(sum(cm,2),1,size(cm,2)) ) );
        end
    end
    
    clsyfyr(s).funcopt = param;
    clsyfyr(s).clsyfyropt = clsyfyropt;
    clsyfyr(s).cvopt = innercvparam;
    clsyfyr(s).numfolds = numfolds;
    clsyfyr(s).numcvfolds = numcvfolds;
end

if strcmp(param.mode,'train')
    outputs = {clsyfyr model};
else
    outputs = {clsyfyr};
end

end

function perf = gridsearch(features,labels,type,cvopt,clsyfyropt,hyperparam)

switch type
    case 'knn'
        Nvals = hyperparam{1};
        perf = zeros(size(Nvals));
        for n = 1:length(Nvals)
            model = fitcecoc(features,labels,cvopt{:}, ...
                'Learners',templateKNN(clsyfyropt{:},'NumNeighbors',Nvals(n)));
            perf(n) = getperf(model,labels);
        end
        
    case 'svm-linear'
        Cvals = hyperparam{1};
        perf = zeros(length(Cvals));
        for c = 1:length(Cvals)
            model = fitcecoc(features,labels,cvopt{:}, ...
                'Learners',templateSVM(clsyfyropt{:},'BoxConstraint',Cvals(c)));
            perf(c) = getperf(model,labels);
        end
        
    case 'svm-rbf'
        Cvals = hyperparam{1};
        Kvals = hyperparam{2};
        perf = zeros(length(Cvals),length(Kvals));
        for c = 1:length(Cvals)
            for k = 1:length(Kvals)
                model = fitcecoc(features,labels,cvopt{:}, ...
                    'Learners',templateSVM(clsyfyropt{:},'KernelFunction','RBF',...
                    'BoxConstraint',Cvals(c),'KernelScale',Kvals(k)));
                perf(c,k) = getperf(model,labels);
            end
        end
        
    case 'tree'
        Lvals = hyperparam{1};
        perf = zeros(size(Lvals));
        for l = 1:length(Lvals)
            model = fitcecoc(features,labels,cvopt{:}, ...
                'Learners',templateTree('MinLeafSize',Lvals(l)));
            perf(l) = getperf(model,labels);
        end
end
end

function [perf,cm] = getperf(model,labels)
predlabels = kfoldPredict(model);
cm = confusionmat(labels,predlabels,'order',unique(labels));
perf = mean( diag( cm ./ repmat(sum(cm,2),1,size(cm,2)) ) );
end

function data = applypca(data,pcacoeff)
if ~isempty(pcacoeff)
    data = data * pcacoeff;
end
end