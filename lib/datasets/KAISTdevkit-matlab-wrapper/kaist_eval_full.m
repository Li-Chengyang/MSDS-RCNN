% kaist_eval_full
% Day and night
% day
% night
% all
% Scale
% near                 [ 115         ]
% medium         [ 45    115 ]
% far                    [           45  ]
% Occlusion
% no
% partial
% heavy
function kaist_eval_full(dtDir, gtDir, reval, writeRes)

if nargin < 3, reval = true; end
if nargin < 4, writeRes = true; end

sepPos = find(dtDir=='\' | dtDir=='/');
if length(dtDir) == sepPos(end)
    sepPos(end) = []; 
    dtDir(end) = [];
end
tname = dtDir(sepPos(end)+1:end);

bbsNms = aggreg_dets(dtDir, reval, tname);

exps = {
  'Reasonable-all',       'test-all',       [55 inf],    {{'none','partial'}}
  'Reasonable-day',    'test-day',    [55 inf],    {{'none','partial'}}
  'Reasonable-night', 'test-night', [55 inf],    {{'none','partial'}}
  'Scale=near',              'test-all',       [115 inf], {{'none'}}
  'Scale=medium',      'test-all',        [45 115],   {{'none'}}
  'Scale=far',                  'test-all',       [1 45],   {{'none'}}
  'Occ=none',               'test-all',       [1 inf],      {{'none'}}
  'Occ=partial',             'test-all',       [1 inf],      {{'partial'}}
  'Occ=heavy',              'test-all',        [1 inf],     {{'heavy'}}
  };

res = [];

for ie = 1:9
    res = run_exp(res, exps(ie,:), gtDir, bbsNms);
end

if writeRes
    save(fullfile(dtDir, '..', ['res' tname(4:end) '.mat']), 'res');
    fprintf('Results saved.\n');
end

end

function bbsNms = aggreg_dets(dtDir, reval, tname)
% return aggregated files
% bbsNm.test-all
for cond = [{'test-all'}, {'test-day'}, {'test-night'}]
    desName = [tname '-' cond{1} '.txt'];
    desName = fullfile(dtDir, '..', desName);
    bbsNms.(sprintf('%s', strrep(cond{1}, '-', '_'))) = desName;
    if exist(desName, 'file') && ~reval
        continue;
    end
    switch cond{1}
        case 'test-all'
            setIds = 6:11;
            skip = 20;
            vidIds = {0:4 0:2 0:2 0 0:1 0:1};
        case 'test-day'
            setIds = 6:8; 
            skip = 20;
            vidIds = {0:4 0:2 0:2};
        case 'test-night'
            setIds = 9:11;
            skip = 20;
            vidIds = {0 0:1 0:1};
    end
    fidA = fopen(desName, 'w');
    num = 0;
    for s=1:length(setIds)
        for v=1:length(vidIds{s})
            for i=skip-1:skip:99999
                detName = sprintf('set%02d_V%03d_I%05d.txt', setIds(s), vidIds{s}(v), i); 
                detName = fullfile(dtDir, detName);
                if ~exist(detName, 'file')
                    continue;
                end
                num = num + 1;
                [~, x1, y1, x2, y2, score] = textread(detName, '%s %f %f %f %f %f');
                for j = 1:length(score)
                    fprintf(fidA, '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n', num, x1(j)+1, y1(j)+1, x2(j)-x1(j), y2(j)-y1(j), score(j));
                end
            end
        end
    end
    fclose(fidA);
end

end

function res = run_exp(res, iexp, gtDir, bbsNms)

thr = .5;
mul = 0;
ref = 10.^(-2:.25:0);
pLoad0={'lbls',{'person'},'ilbls',{'people','person?','cyclist'}};
pLoad = [pLoad0, 'hRng',iexp{3}, 'vType',iexp{4},'xRng',[5 635],'yRng',[5 507]];

res(end+1).name = iexp{1};

bbsNm = bbsNms.(sprintf('%s',strrep(iexp{2},'-','_')));
% original annotations
annoDir = fullfile(gtDir,iexp{2},'annotations');
[gt,dt] = bbGt('loadAll',annoDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,ref);
miss_ori=exp(mean(log(max(1e-10,1-miss))));
roc_ori=[score fp tp];

res(end).ori_miss = miss;
res(end).ori_mr = miss_ori;
res(end).roc = roc_ori;

% improved annotations
annoDir = fullfile(gtDir,iexp{2}, 'annotations_KAIST_test_set');
[gt,dt] = bbGt('loadAll',annoDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,ref);
miss_imp=exp(mean(log(max(1e-10,1-miss))));
roc_imp=[score fp tp];

res(end).imp_miss = miss;
res(end).imp_mr = miss_imp;
res(end).imp_roc = roc_imp;

fprintf('%-30s \t log-average miss rate = %02.2f%% (%02.2f%%) recall = %02.2f%% (%02.2f%%)\n', iexp{1}, miss_ori*100, miss_imp*100, roc_ori(end, 3)*100, roc_imp(end, 3)*100);

end
