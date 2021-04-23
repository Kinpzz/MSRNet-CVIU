%
%  prop2prob: transfer the bouding boxes of salient proposals into K
%             (instance number)+1 channels probability map.
%
%  INPUT:
%  - bbox_props   : Bounding boxes of salient proposals ( [4,bbox_numbers] )
%                   [x1;y1;x2;y2] (coorfinate of left-down and right-up)
%                   every bbox occupy one column
%  - salient_mask : Salient region mask ( uint8, [height, width] )
%                   salient pixels value is 255, background value is 0
%
%  OUTPUT:
%  - prob         : probability map with K+1 channels, channel of top K
%                   channels represents the instance's salient propability,
%                   the K+1 channel repesents the background
%
function [ prob ] = prop2prob( bbox_props, salient_mask )

if ~isa(salient_mask(1,1), 'uint8')
    salient_mask = im2uint8(salient_mask);
end
bbox_props = round(bbox_props);

% transfer salient mask to bool map
salient_region = salient_mask > 127;

instance_num = size(bbox_props,2);
[height, width] = size(salient_mask);
% initial probability map
prob = zeros([height,width,instance_num+1]);

% bbox
bbox_map = zeros([height,width, instance_num]);
bbox_count_map = zeros([height,width]);
% instance_channel
for i = 1:instance_num
    x1 = bbox_props(1,i);
    y1 = bbox_props(2,i);
    x2 = bbox_props(3,i);
    y2 = bbox_props(4,i);
    bbox_map(y1:y2,x1:x2,i) = 1;
    bbox_count_map(y1:y2,x1:x2) = bbox_count_map(y1:y2,x1:x2) + 1;
end
for i = 1:instance_num
    proposal_region = bbox_map(:,:,i) > 0;
    % salient region covered with proposal(i)
    area_a = salient_region & proposal_region;
    prob_channel = zeros([height, width]);
    prob_channel(area_a) = 1 ./ bbox_count_map(area_a);
    % salient region covered without any proposals
    area_b = salient_region & (bbox_count_map == 0);
    prob_channel(area_b) = 1 / (instance_num + 1);
    % bg covered with proposal(i)
    area_c = ~salient_region & proposal_region;
    prob_channel(area_c) = 1 ./ (bbox_count_map(area_c) + 1);
    prob(:,:,i) = prob_channel;
end

% bg channel
area_bg = ~salient_region;
prob_channel = zeros([height, width]);
prob_channel(area_bg) = 1 ./ (bbox_count_map(area_bg) + 1);
prob(:,:,instance_num+1) = prob_channel;

end

