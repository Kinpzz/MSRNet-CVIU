function cfg = config()
cfg.pipeline_root = fileparts(which(mfilename));
cfg.crf_path = fullfile(cfg.pipeline_root, 'meanfield-matlab');
cfg.saliency_threshold = 127;
cfg.contour_threshold = 200;

% proposals settings
cfg.filter_thresh = 0.05;
cfg.extend_bbox_percent = 0.05;
cfg.min_extend_bbox_pixels = 5;
cfg.align_thres = 5;
cfg.NMS_threshold=0.7;

% crf settings.
cfg.gaussian_x_stddev = 3;
cfg.gaussian_y_stddev = 3;
cfg.gaussian_weight = 3;

cfg.bilateral_x_stddev = 49;
cfg.bilateral_y_stddev = 49;
cfg.bilateral_r_stddev = 5;
cfg.bilateral_g_stddev = 5;
cfg.bilateral_b_stddev = 5;
cfg.bilateral_weight = 4;

% visual settings
cfg.color_map = [0,0,0;0,0,125;255,0,0;0,85,0;170,0,51;148,0,211;...
    218,165,32;165,42,42;0,0,170;255,20,147;255,255,210]/255;

addpath(cfg.crf_path);
addpath(fullfile(cfg.pipeline_root, 'proposals'));

end