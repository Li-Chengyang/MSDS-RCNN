function demo_test()

%% specify path of detections
dtDir = 'path_to_detections/detections/MSDS_sanitized/det';

%% specify path of groundtruth annotaions
gtDir = 'path_to_dataset/kaist';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, false, true);

end
