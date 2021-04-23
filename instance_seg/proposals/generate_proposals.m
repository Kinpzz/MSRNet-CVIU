%
% generate_proposals : generate propsals by using saliency map and contour
%
% INPUT:
% - saliency_map : saliency map
% - contour: contour of the same image with saliency map
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
function [bboxes] = generate_proposals(saliency_map, contour, config)
    % init
    bboxes = [];
    sub_bboxes = [];
    % compute connected area of the image
    imsiz = size(saliency_map);
    saliency_region = zeros(imsiz);
    saliency_region(saliency_map > config.saliency_threshold) = 1;
    region = bwareaopen(saliency_region,10);
    %region = bwlabel(region);
    stats = regionprops(region);
    Ar = cat(1, stats.Area);
    % filter small objects to avoid noise
    del_stats_index = [];
    for temp = 1:size(stats)
       if (stats(temp).Area < config.filter_thresh * sum(Ar))
           del_stats_index = [del_stats_index; temp];
       end
    end
    stats(del_stats_index) = [];
    for i = 1:size(stats)
       % compute sub connected area for each connected area
       bbox = ceil(stats(i).BoundingBox);
       bbox([3,4]) = bbox([3,4]) + bbox([1,2])-1;
       bbox(1) = max(1,bbox(1));
       bbox(2) = max(1,bbox(2));
       bbox(3) = min(imsiz(2),bbox(3));
       bbox(4) = min(imsiz(1),bbox(4));
       sub_region = zeros(imsiz);
       sub_region(bbox(2):bbox(4), bbox(1):bbox(3)) = saliency_region(bbox(2):bbox(4), bbox(1):bbox(3));
       sub_contour = contour(bbox(2):bbox(4), bbox(1):bbox(3));
       % if this sub area is also detected by contour, else this won't be a proposal
       if sum(sub_contour(:)) < 0.1*sum(sub_region(:))
           continue;
       end
       sub_region(contour > config.contour_threshold) = 0;
       sub_region = bwareaopen(sub_region, 5);
       %sub_region = bwlabel(sub_region);
       sub_stats = regionprops(sub_region);
       % remove small object
       Ar = cat(1, sub_stats.Area);
       del_sub_stats_index = [];
       for temp = 1:size(sub_stats)
           if (sub_stats(temp).Area < config.filter_thresh * sum(Ar(:)))
               del_sub_stats_index = [del_sub_stats_index; temp];
           end
       end
       sub_stats(del_sub_stats_index) = [];
       % if this area has not sub connected area, this area will be a poprosal
       for j = 1:size(sub_stats)
           delete_sub = [];
           sub_bbox = ceil(sub_stats(j).BoundingBox);
           sub_bbox([3,4]) = sub_bbox([3,4]) + sub_bbox([1,2])-1;
           for t = j+1:size(sub_stats)
               sub_bbox_2 = ceil(sub_stats(t).BoundingBox);
               sub_bbox_2([3,4]) = sub_bbox_2([3,4]) + sub_bbox_2([1,2])-1;
               if compute_IoU(sub_bbox, sub_bbox_2) > config.NMS_threshold
                   delete_sub = [delete_sub; t];
               end
           end
       end
       sub_stats(delete_sub) = [];
       if size(sub_stats) <= 1
           bboxes = [bboxes; bbox];
           continue;
       end
       for j = 1:size(sub_stats)
           sub_bbox = ceil(sub_stats(j).BoundingBox);
           sub_bbox([3,4]) = sub_bbox([3,4]) + sub_bbox([1,2])-1;
           % make up the lose of contour
           height_add = max(ceil(sub_stats(j).BoundingBox(3) * config.extend_bbox_percent),config.min_extend_bbox_pixels);
           width_add = max(ceil(sub_stats(j).BoundingBox(4) * config.extend_bbox_percent),config.min_extend_bbox_pixels);
           sub_bbox = sub_bbox + [-width_add, -height_add, width_add, height_add];
           sub_bbox(1) = max(1,sub_bbox(1));
           sub_bbox(2) = max(1,sub_bbox(2));
           sub_bbox(3) = min(imsiz(2),sub_bbox(3));
           sub_bbox(4) = min(imsiz(1),sub_bbox(4));
           % compare with uncle node
           cont_flag = 0;
           for temp = 1:size(bboxes)
               if compute_IoU(sub_bbox,bboxes(temp,:)) > config.NMS_threshold
                   cont_flag = 1;
                   break;
               end
           end
           if cont_flag == 1
               continue;
           end
           % align to father bbox
           if abs(sub_bbox(1) - bbox(1)) < config.align_thres
               sub_bbox(1) = bbox(1);
           end
           if abs(sub_bbox(2) - bbox(2)) < config.align_thres
               sub_bbox(2) = bbox(2);
           end
           if abs(bbox(3) - sub_bbox(3)) < config.align_thres
               sub_bbox(3) = bbox(3);
           end
           if abs(bbox(4) - sub_bbox(4)) < config.align_thres
               sub_bbox(4) = bbox(4);
           end
           sub_bboxes = [sub_bboxes; sub_bbox];
       end
    end
    bboxes = [bboxes; sub_bboxes];
    bboxes = bboxes';
end

% compute iou of two bboxes
% input bboxes: (y1,x1,y2,x2) <-->(row1,col1,row2,col2)
function iou = compute_IoU(bbox1,bbox2)
    % coordinates of intersection rectangle
    y1 = max(bbox1(1), bbox2(1));
    x1 = max(bbox1(2), bbox2(2));
    y2 = min(bbox1(3), bbox2(3));
    x2 = min(bbox1(4), bbox2(4));
    inter_area = (x2 - x1 + 1) * (y2 - y1 + 1);
    bbox1_area = (bbox1(3) - bbox1(1) + 1) * (bbox1(4) - bbox1(2) + 1);
    bbox2_area = (bbox2(3) - bbox2(1) + 1) * (bbox2(4) - bbox2(2) + 1);
    iou = inter_area / (bbox1_area + bbox2_area - inter_area);
end