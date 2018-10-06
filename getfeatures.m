function features = getfeatures(listname,measure,bandidx,varargin)

param = finputcheck(varargin, {
    'changroup', 'string', [], ''; ...
    'changroup2', 'string', [], ''; ...
    'trange', 'real', [], []; ...    
    });

loadpaths

load(sprintf('%sgroupdata_%s.mat',filepath,listname),'bandpower','subjlist');

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
elseif strcmpi(measure,'wpli')
    load(sprintf('%sgroupdata_%s.mat',filepath,listname),'allcoh');
    ind_upper = find(triu(ones(size(allcoh,3),size(allcoh,4)),1))';
    features = allcoh(:,bandidx,ind_upper);    
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
    features = cellfun(@(x) str2num(x(2:3)), subjlist.name);
else
    load(sprintf('%s/groupdata_%s.mat',filepath,listname),'graph','tvals');
    
    %round down to 3 decimal places
    precision = 3;
    threshidx = ismember(round(tvals * 10^precision),round(param.trange * 10^precision));
    if ~all(round(tvals(threshidx) * 10^precision) == round(param.trange * 10^precision))
        error('getfeatures: some requested thresholds not found!');
    end
    
    m = strcmpi(measure,graph(:,1));
    
    if strcmpi(measure,'mutual information')
        features = mean(graph{m,weiorbin}(:,:,bandidx,threshidx),4);
        features = permute(features,[1 3 2]);
    else
        features = squeeze(graph{m,weiorbin}(:,bandidx,:,:));
        features = features(:,threshidx,:);
    end
end
