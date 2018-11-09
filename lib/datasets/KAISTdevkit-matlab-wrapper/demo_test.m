function demo_test()

%% specify path of detections
% dtDir = '/home/lcy/Research/MyCodes/pd_res/results_BMVC16_Liu_et_al/KASIT_halfway_fusion/det';
% dtDir = '/home/lcy/Research/MyCodes/results_BMVC16_Liu_et_al/KASIT_late_fusion/det';
% dtDir = '/home/lcy/Research/MyCodes/results_BMVC16_Liu_et_al/KASIT_score_fusion/det';
% dtDir = '/home/lcy/datasets/pedestrian/kaist/piotr-toolbox-3.40_modified/detector/models/det';
dtDir = '/home/lcy/MyCodes/MSDS-RCNN/output/vgg16/kaist_test-all_multi_50np/default/vgg16_msds_rcnn_iter_45606/det';

%% specify path of groundtruth annotaions
gtDir = '/media/lcy/ssd1/kaist';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, false, true);

end
