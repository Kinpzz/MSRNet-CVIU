%
% instance_sementation: run instance segmentation pipeline
% INPUT:
% - I: input image
% - saliency_map : saliency map of the input image
% - contour: contour of the input image
% - config
% OUTPUT:
% - bboxes: proposals of the input image. Each proposal is an array
%           of bounding boxes, which contains four coordinates:
%           left-top corner (y1, x1), right-bottom corner (y2, x2)
%           (y1,x1) ------  (y1,x2)
%              |               |
%              |               |
%           (y2,x1) ------  (y2,x2)
%          size: (n, 4), n stands for proposals number

function [seg, bboxes] = instance_segmentation(I, salient_mask, contour, config)

% generate proposals

bboxes = generate_proposals(salient_mask, contour, config);

% crf
if (isempty(bboxes))
    unary = zeros(size(salient_mask,1), size(salient_mask, 2), 2);
    unary_channel = zeros(size(salient_mask));
    unary_channel(salient_mask > config.saliency_threshold) = 0.5;
    unary(:,:, 1) = unary_channel;
    unary_channel = zeros(size(salient_mask));
    unary_channel(salient_mask <= config.saliency_threshold) = 0.5;
    unary(:,:, 2) = unary_channel;
else
    unary = prop2prob_by_mask(bboxes, salient_mask ,salient_mask);
end
unary = -log(single(unary));

D = Densecrf(I,unary);

% crf parameters settings.
D.gaussian_x_stddev = config.gaussian_x_stddev;
D.gaussian_y_stddev = config.gaussian_y_stddev;
D.gaussian_weight = config.gaussian_weight;

D.bilateral_x_stddev = config.bilateral_x_stddev;
D.bilateral_y_stddev = config.bilateral_y_stddev;
D.bilateral_r_stddev = config.bilateral_r_stddev;
D.bilateral_g_stddev = config.bilateral_g_stddev;
D.bilateral_b_stddev = config.bilateral_b_stddev;
D.bilateral_weight = config.bilateral_weight;

seg=uint8(D.mean_field)-1;

end