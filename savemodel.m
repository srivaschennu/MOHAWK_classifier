function savemodel(clsyfyrlist,varargin)

param = finputcheck(varargin, {
    'group', 'string', [], 'crsdiagwithcmd'; ...
    });

loadpaths

fprintf('Loading classifiers:');
for c = 1:length(clsyfyrlist)
    fprintf(' %s',clsyfyrlist{c});
    
    if c == 1
        load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,clsyfyrlist{c}),'output1','output2','clsyfyrinfo');
        clsyfyr = vertcat(output1{:});
        model = output2;
    elseif c > 1
        nextclsyfyr = load(sprintf('%sclsyfyr_%s_%s.mat',filepath,param.group,clsyfyrlist{c}),'output1','clsyfyrinfo');
        clsyfyr = cat(1,clsyfyr,vertcat(nextclsyfyr.output1{:}));
        model = cat(1,model,output2);
        clsyfyrinfo.clsyfyrparam = cat(1,clsyfyrinfo.clsyfyrparam,nextclsyfyr.clsyfyrinfo.clsyfyrparam);
    end
end
fprintf('\n');

load(sprintf('%scombclassifier.mat', filepath), 'allbel');
clsyfyrinfo.groupnames{2} = 'MCS';

fprintf('Saving trained model to %scombmodel.mat.\n', filepath);
save(sprintf('%scombmodel.mat', filepath), 'clsyfyr', 'model', 'clsyfyrinfo', 'allbel');