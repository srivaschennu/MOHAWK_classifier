function features = getfeatures(listname,measure,bandidx,varargin)

param = finputcheck(varargin, {
    'changroup', 'string', [], ''; ...
    'changroup2', 'string', [], ''; ...
    'trange', 'real', [], []; ...    
    });

loadpaths
loadsubj

load(sprintf('%sgroupdata_%s.mat',filepath,listname),'subjlist','bandpower');

loadcovariates
changroups

switch size(bandpower,3)
    case 91
        load sortedlocs_91.mat
        param.changroup = 'allchan_91';
        param.changroup2 = 'allchan_91';
    case 173
        load sortedlocs_173.mat
        param.changroup = 'allchan_173';
        param.changroup2 = 'allchan_173';
end


weiorbin = 2;

if strcmpi(measure,'power')
    load(sprintf('%sgroupdata_%s.mat',filepath,listname),'bandpower');
    features = bandpower(:,bandidx,ismember({sortedlocs.labels},eval(param.changroup))) * 100;
elseif strcmpi(measure,'specent')
    load(sprintf('%sgroupdata_%s.mat',filepath,listname),'specent');
    features = specent(:,ismember({sortedlocs.labels},eval(param.changroup)));
elseif strcmpi(measure,'median')
    load(sprintf('%sgroupdata_%s.mat',filepath,listname),'allcoh');
    features = median(allcoh(:,bandidx,ismember({sortedlocs.labels},eval(param.changroup)),ismember({sortedlocs.labels},eval(param.changroup2))),4);
elseif strcmpi(measure,'mean')
    load(sprintf('%sgroupdata_%s.mat',filepath,listname),'allcoh');
    features = mean(allcoh(:,bandidx,ismember({sortedlocs.labels},eval(param.changroup)),ismember({sortedlocs.labels},eval(param.changroup2))),4);
elseif strcmpi(measure,'refdiag')
    features = refdiag;
elseif strcmpi(measure,'demo')
    features = [etiology daysonset];
elseif strcmpi(measure,'subjnum')
    features = cellfun(@(x) str2num(x(2:3)), subjlist(:,1));
else
    load(sprintf('%s/groupdata_%s.mat',filepath,listname),'graph','tvals');
    m = strcmpi(measure,graph(:,1));
    features = squeeze(graph{m,weiorbin}(:,bandidx,:,:));

    %round down to 3 decimal places
    precision = 3;
    threshidx = ismember(round(tvals * 10^precision),round(param.trange * 10^precision));
    if ~all(round(tvals(threshidx) * 10^precision) == round(param.trange * 10^precision))
        error('getfeatures: some requested thresholds not found!');
    end
    features = features(:,threshidx,:);
end
