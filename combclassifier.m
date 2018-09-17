function clsyfyrorder = combclassifier(clsyfyrlist,varargin)

param = finputcheck(varargin, {
    'mode', 'string', {'eval','test'}, 'eval'; ...
    'testlist', 'cell', {}, {}; ...
    'group', 'string', [], 'crsdiagwithcmd'; ...
    'nclsyfyrs', 'real', [], []; ...
    });

loadpaths

colorlist = [
    0 0.0 0.5
    0 0.5 0
    0.5 0.0 0
    0   0.5 0.5
    0.5 0   0.5
    0.5 0.5 0
    ];

facecolorlist = [
    0.75  0.75 1
    0.25 1 0.25
    1 0.75 0.75
    0.75 1 1
    1 0.75 1
    1 1 0.5
    ];

fontsize = 20;

fprintf('Loading classifiers:');
for c = 1:length(clsyfyrlist)
    fprintf(' %s',clsyfyrlist{c});
    if strcmp(param.mode,'test')
        fprintf(' %s',param.testlist{c});
    end
    
    if c == 1
        load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,clsyfyrlist{c}),'output1','clsyfyrinfo');
        clsyfyr = vertcat(output1{:});
        clsyfyr = clsyfyr(:,1);
        if strcmp(param.mode,'test')
            load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,param.testlist{c}),'output1');
            testres = vertcat(output1{:});
        end
    elseif c > 1
        nextclsyfyr = load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,clsyfyrlist{c}),'output1','clsyfyrinfo');
        clsyfyr = cat(1,clsyfyr,vertcat(nextclsyfyr.output1{:}));
        clsyfyrinfo.clsyfyrparam = cat(1,clsyfyrinfo.clsyfyrparam,nextclsyfyr.clsyfyrinfo.clsyfyrparam);
        if strcmp(param.mode,'test')
            nextclsyfyr = load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,param.testlist{c}),'output1');
            testres = cat(1,testres,vertcat(nextclsyfyr.output1{:}));
        end
    end
end
fprintf('\n');

if isempty(param.nclsyfyrs)
    param.nclsyfyrs = length(clsyfyr);
end

numgroups = length(clsyfyrinfo.groups);
if strcmp(param.mode,'eval')
    truelabels = clsyfyr(1).truelabels;
    numfolds = clsyfyr(1).numfolds;
    numcvfolds = clsyfyr(1).numcvfolds;
    numruns = numfolds/numcvfolds;
    groupnames = clsyfyrinfo.groupnames;
elseif strcmp(param.mode,'test')
    truelabels = testres(1).truelabels;
    numruns = 1;
    numcvfolds = 1;
    groupnames = clsyfyrinfo.groupnames;
end

for c = 1:length(clsyfyr)
    clsyfyr(c).cm = round(clsyfyr(c).cm * 100 ./ repmat(sum(clsyfyr(c).cm,2),1,size(clsyfyr(c).cm,2),1));
    clsyfyr(c).cm = clsyfyr(c).cm + eps;
    clsyfyr(c).cm = clsyfyr(c).cm ./ repmat(sum(clsyfyr(c).cm,1),size(clsyfyr(c).cm,1),1,1);
end

combperf = NaN(param.nclsyfyrs,numruns);
testperf = NaN(param.nclsyfyrs,numruns);
combclassperf = NaN(param.nclsyfyrs,numgroups,numruns);
confmat = NaN(numgroups,numgroups,param.nclsyfyrs,numruns);

[trainperf,perfsort] = sort(arrayfun(@(x) mean(x.perf),clsyfyr),'descend');
trainperf = trainperf(1:param.nclsyfyrs);
clsyfyrorder = clsyfyrinfo.clsyfyrparam(perfsort,:);
allbel = ones(length(truelabels),numgroups,param.nclsyfyrs,numruns);

fprintf('CV run');
for c = 1:numruns
    fprintf(' %d',c);
    bel = ones(length(truelabels),numgroups);
    for k = 1:param.nclsyfyrs
        if strcmp(param.mode,'eval')
            testperf(k,c) = clsyfyr(perfsort(k)).testperf(c);
        end
        for f = (c-1)*numcvfolds+1:c*numcvfolds
            if strcmp(param.mode,'eval')
                if sum(~isnan(clsyfyr(perfsort(k)).predlabels(:,f)) ~= ~isnan(clsyfyr(1).predlabels(:,f))) ~= 0
                    error('holdout mismatch');
                end
                thisfold = find(~isnan(clsyfyr(perfsort(k)).predlabels(:,f)));
                thispred = clsyfyr(perfsort(k)).predlabels(thisfold,f);
            elseif strcmp(param.mode,'test')
                thisfold = find(~isnan(testres(perfsort(k)).predlabels(:,f)));
                thispred = testres(perfsort(k)).predlabels(:,f);
            end
            
            for p = 1:size(thisfold,1)
                bel(thisfold(p),:) = bel(thisfold(p),:) .* clsyfyr(perfsort(k)).cm(:,thispred(p)+1,f)';
            end
        end
        bel = bel ./ repmat(sum(bel,2),1,size(bel,2));
%         [roc(k,c).fpr, roc(k,c).tpr] = perfcurve(truelabels,bel(:,2),max(truelabels));
        predlabels = round(sum(bel .* repmat(1:size(bel,2),size(bel,1),1),2));
        predlabels = predlabels - 1;

        cm = confusionmat(truelabels,predlabels);
        confmat(:,:,k,c) = cm;
        
        normcm = cm ./ repmat(sum(cm,2),1,size(cm,2));
        combperf(k,c) = mean(diag(normcm));
        combclassperf(k,:,c) = diag(normcm);
        
        allbel(:,:,k,c) = bel;
    end
end
fprintf('\nDone.\n');

fig_h = figure('Color','white','Name',cell2mat(clsyfyrlist));
% fig_h.Position(3) = fig_h.Position(3) * 1.5;
hold all

combperf = combperf * 100; testperf = testperf * 100; trainperf = trainperf * 100; combclassperf = combclassperf * 100;


combperf = mean(combperf,2);
testperf = mean(testperf,2);

plot(combperf,'LineWidth',2,'Color','black','DisplayName','Combined');
for g = 1:numgroups
    plot(mean(combclassperf(:,g,:),3),'LineWidth',1,'LineStyle','-.','Color',colorlist(g,:),'DisplayName',groupnames{g});
end
plot(trainperf,'LineStyle','--','LineWidth',1.5,'Color',colorlist(g+1,:),'DisplayName','Train');
plot(testperf,'LineStyle','--','LineWidth',1.5,'Color',colorlist(g+2,:),'DisplayName','Test');

plot([1 param.nclsyfyrs],[100/numgroups 100/numgroups],'Color','blue',...
    'LineStyle',':','LineWidth',1.5,'DisplayName','Chance');

legend('Location','SouthEast');

[~,bestk] = max(combperf);
plot([1 bestk],[combperf(bestk) combperf(bestk)],'LineStyle',':','LineWidth',1.5,'Color','black');
[~,besttest] = max(testperf);
plot([1 besttest],[testperf(besttest) testperf(besttest)],'LineStyle',':','LineWidth',1.5,'Color','black');

xlim([1 param.nclsyfyrs]);
ylim([0 100]);
set(gca,'FontName','Helvetica','FontSize',fontsize);
xlabel('Number of classifiers','FontName','Helvetica','FontSize',fontsize);
ylabel('Accuracy','FontName','Helvetica','FontSize',fontsize);

[~,bestk] = max(combperf);
plot([bestk bestk],ylim,...
    'LineStyle','-','LineWidth',1.5,'Color','red','DisplayName','Peak accuracy');

% figure('Color','white');
% plot(roc(bestk,1).fpr,roc(bestk,1).tpr,'LineWidth',2);
% set(gca,'FontName','Helvetica','FontSize',fontsize);
% xlabel('False Positive Rate','FontName','Helvetica','FontSize',fontsize);
% ylabel('True Positive Rate','FontName','Helvetica','FontSize',fontsize);

plotconfusionmat(sum(confmat(:,:,bestk,:),4),groupnames);
plotconfusionmat(sum(clsyfyr(perfsort(1)).cm,3),groupnames);

figure('Color','white');
figpos = get(gcf,'Position');
figpos(3) = figpos(3)*1/2;
set(gcf,'Position',figpos);

boxh = notBoxPlot(mean(allbel(:,2,bestk,:),4),truelabels+1,0.5,'patch',ones(size(truelabels,1),1));
for h = 1:length(boxh)
    set(boxh(h).data,'Color',colorlist(h,:),'MarkerFaceColor',facecolorlist(h,:))
end
set(gca,'FontName','Helvetica','FontSize',fontsize);
set(gca,'XLim',[0.5 numgroups+0.5], 'YLim',[0 1], 'XTick',1:numgroups,...
        'XTickLabel',groupnames','FontName','Helvetica','FontSize',fontsize);
ylabel('Probability','FontName','Helvetica','FontSize',fontsize);

allbel = mean(allbel(:,2,bestk,:),4);
perfsort = perfsort(1:bestk);
save(sprintf('%s/combclsyfyr_%s.mat', filepath, strtok(clsyfyrlist{1},'_')), 'clsyfyrinfo','perfsort','allbel','truelabels');