function plotnbs(listname,bandidx,varargin)

param = finputcheck(varargin, {
    'xlim', 'real', [], []; ...
    'ylim', 'real', [], []; ...
    'legendlocation', 'string', [], 'Best'; ...
    });

fontname = 'Helvetica';
fontsize = 28;

loadpaths

bands = {
    'delta'
    'theta'
    'alpha'
    'beta'
    'gamma'
    };

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

load(sprintf('%s/%s_%s_stats.mat',filepath,listname,bands{bandidx}));

if n_nets == 0
    fprintf('No significant network components found.\n');
    return
end

load(sprintf('%s/groupdata_%s.mat',filepath,listname));

corrmat = zeros(size(allcoh,3),size(allcoh,4));
ind_upper = find(triu(ones(size(allcoh,3),size(allcoh,4)),1))';
corrmat(ind_upper) = test_stat(1,:);
corrmat(~netmask{1}) = 0;
sumcorr = sum(corrmat(logical(netmask{1})));
corrmat = triu(corrmat,1)+triu(corrmat,1)';

fprintf('Cluster t-statistic = %.2f, p-value = %.4f.\n', sumcorr, netpval);

%% plot 3d graph
plotgraph3d(corrmat,'plotqt',0,'arcs','strength','lhfactor',1.25);

set(gcf,'Name',sprintf('group %s: %s band',listname,bands{bandidx}));
camva(8);
camtarget([-9.7975  -28.8277   41.8981]);
campos([-1.7547    1.7161    1.4666]*1000);
camzoom(1.2);
set(gcf,'InvertHardCopy','off');
print(gcf,sprintf('figures/corr3d_%s_%s.tif',listname,bands{bandidx}),'-dtiff','-r200');
close(gcf);

%% correlate with age

% loadcovariates
% 
% testdata = mean(allcoh(:,bandidx,logical(netmask{1})),3);
% 
% datatable = sortrows(cat(2,testdata,age,gender));
% mdl = fitlm(datatable(:,1),datatable(:,2),'RobustOpts','on');
% pointsize = 50;
% figure('Color','white');
% hold all
% 
% legendoff(scatter(datatable(:,1), ...
%     datatable(:,2),pointsize,...
%     colorlist(1,:),'MarkerFaceColor',facecolorlist(1,:)));
% 
% b = mdl.Coefficients.Estimate;
% plot(datatable(:,1),b(1)+b(2)*datatable(:,1),'--','Color','black','LineWidth',1, ...
%     'Display',sprintf('\\R^2 = %.2f, p = %.3f',mdl.Rsquared.Adjusted,netpval,mdl.Coefficients.pValue(2)));
% 
% set(gca,'FontName',fontname,'FontSize',fontsize);
% if ~isempty(param.ylim)
%     set(gca,'YLim',param.ylim);
% end
% if ~isempty(param.xlim)
%     set(gca,'XLim',param.xlim);
% end
% 
% xlabel('Median dwPLI','FontName',fontname,'FontSize',fontsize);
% ylabel('Age','FontName',fontname,'FontSize',fontsize);
% 
% leg_h = legend('show');
% set(leg_h,'Location',param.legendlocation);
% txt_h = findobj(leg_h,'type','text');
% set(txt_h,'FontSize',fontsize-4)
% legend('boxoff');
% 
% export_fig(gcf,sprintf('figures/corr_%s.eps',bands{bandidx}));
% close(gcf);