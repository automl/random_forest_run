function fanova_test2
rand('twister', 1234); % for deterministic tests.

al_opts = get_rf_default_options;
al_opts.Splitmin_init = 1;
al_opts.nSub = 1;
al_opts.kappa_max = inf; %not used.
al_opts.logModel = 0;
al_opts.storeDataInLeaves = 0;
import ca.ubc.cs.beta.models.rf.*;
al_opts.modelType = 'javarf';

n=1000;
dim = 10;
X = rand(n,dim);
y = 100 * X(:,1) + 10*X(:,2);
Xtrain = X;
% ytrain = log10(y);
ytrain = y;

%=== Learn model.
thetaCatDomains = cell(size(X,2),1);
for i=1:length(thetaCatDomains)
    thetaCatDomains{i} = {};
end
model = learnModel(Xtrain, ytrain, zeros(length(ytrain),1), [], thetaCatDomains, [], [], [], 1, al_opts, {});

nTest = 10;
X = rand(nTest,dim);


% 
% %=== Functional ANOVA to determine main effects and interaction effects.
% Xall = X(:,1);
% for i=2:size(X,2)
%     xnew = repmat(X(:,i)', [size(Xall,1),1]);
%     Xall = repmat(Xall, [size(X,1),1]);
%     Xall = [Xall, xnew(:)];
% end
% 
% [ypreds, yvars] = applyModel(model, Xall, 1, 0, 0);
% f0 = mean(ypreds);
% overall_variance = mean_squared_diff(ypreds-f0)

Xcell = cell(size(X,2), 1);
for j=1:length(Xcell)
    Xcell(j) = {X(:,j)};
end

all_inds = allTrueSubsets(1:size(X,2));
all_inds = sort_by_length(all_inds);


num_tree=1;

%=== Compute total variance by iterating over leaves. Only works for a
%=== single regrerssion tree at a time.
total_variance = RegtreeFanova.getTotalVariance(model.T.Trees(num_tree), Xcell);
overall_variance=total_variance


inds = [];
for i=1:length(all_inds)
    if length(all_inds{i})==1
        inds = [inds; all_inds{i}];
    end
end
for single_idx=1:size(inds,1)
    i=inds(single_idx);
    Xvals = X(:,i);
    %=== Construct cross-product of remaining X.
    rest_ids = setdiff(1:size(X,2), i);
    Xrest = X(:,rest_ids);
    Xcell = cell(size(Xrest,2), 1);
    for j=1:length(Xcell)
        Xcell(j) = {Xrest(:,j)};
    end
    ypreds1 = RandomForest.getMarginal(model.T, Xvals, Xcell, i);
    
    f1(i,:) = ypreds1'-f0;
%    figure;     plot(Xvals, ypreds1, '.');
    tmp = f1(i,:);
    vars(i) = mean_squared_diff(tmp(:));
end
single_vars = vars


vars2 = [];
inds2 = [];
for i=1:length(all_inds)
    if length(all_inds{i})==2
        inds2 = [inds2; all_inds{i}];
    end
end
for idx=1:size(inds2,1)
    i = inds2(idx,1);
    j = inds2(idx,2);
    
    Xvals = -ones(size(X,1)*size(X,1),2);
    count = 1;
    for a=1:size(X,1)
        for b=1:size(X,1)
            Xvals(count,:) = [X(a,i), X(b,j)];
            count = count+1;
        end
    end
    
    Xrest = X(:,setdiff(1:size(X,2), [i,j]));
    Xcell = cell(size(Xrest,2), 1);
    for k=1:length(Xcell)
        Xcell(k) = {Xrest(:,k)};
    end
    
    [ypreds2] = RandomForest.getMarginal(model.T, Xvals, Xcell, [i,j]);
    
    f2 = -ones(size(inds2,1),size(inds2,1),size(X,1),size(X,1));
    count = 1;
    for a=1:size(X,1)
        for b=1:size(X,1)
            f2(i,j,a,b) = ypreds2(count)-f0-f1(i,a)-f1(j,b);
            count = count+1;
        end
    end
    tmp = f2(i,j,:,:);
    vars2(idx) = mean_squared_diff(tmp(:));
end

% vars

inds2
vars2

% lhs = -ones(size(X,1)*size(X,1)*size(X,1),1);
% rhs = -ones(size(X,1)*size(X,1)*size(X,1),1);
% count = 1;
% for a=1:size(X,1)
%     for b=1:size(X,1)
%         for c=1:size(X,1)
%             lhs(count) = ypreds(count)-f0;
%             rhs(count) = f1(1,a) + f1(2,b) + f1(3,c) + f2(1,2,a,b) + f2(1,3,a,c) + f2(2,3,b,c);
%             count = count+1;
%         end
%     end
% end
% orig = mean_squared_diff(lhs)


% lhs = -ones(size(X,1)*size(X,1),1);
% rhs = -ones(size(X,1)*size(X,1),1);
% count = 1;
% for a=1:size(X,1)
%     for b=1:size(X,1)
% %         f0 + f1(1,a) + f1(2,b) + f2(1,2,a,b) - ypreds2(count)
%         f0 + f1(1,a) + f1(2,b) + f2(1,2,a,b) - ypreds2(count);
%         lhs(count) = ypreds2(count)-f0;
%         rhs(count) = f1(1,a) + f1(2,b) + f2(1,2,a,b);
%         count = count+1;
%     end
% end
% orig = mean_squared_diff(lhs)
% mean_squared_diff(rhs)
overall_variance
explained_var = sum(vars)+sum(vars2)
if explained_var > overall_variance+1e-5
    explained_var
end
percentage_of_variance_explained = 100*explained_var/overall_variance
    

function variance = mean_squared_diff(vector)
variance = mean((vector-mean(vector)).^2);
