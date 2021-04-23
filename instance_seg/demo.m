%
% demo: instance salienct object segmentation
%
clear;clc;close all;
cfg = config();
set(0, 'DefaultFigureVisible', 'on')

I = imread('demo_images/demo.jpg');
saliency_map = imread('demo_images/demo_saliency_map.png');
contour = imread('demo_images/demo_contour.png');

contour = (contour > cfg.contour_threshold).* 255;
[seg, proposals] = instance_segmentation(I, saliency_map, contour, cfg);

% display
figure();
subplot(2,3,1), imshow(I), title('input image');
subplot(2,3,2), imshow(saliency_map > cfg.saliency_threshold), title('saliency map');
subplot(2,3,3), imshow(contour > cfg.contour_threshold), title('contour');
subplot(2,3,4), imshow(I), title('box-level proposals')
for i = 1:size(proposals,2)
    rect = proposals(:,i);
    rect(3:4) = rect(3:4)-rect(1:2) +1;
    rectangle('Position',rect,'linewidth',2,'edgecolor',[1 0 0]);
end
subplot(2,3,5), imshow(seg, cfg.color_map), title('instance saliency segmentation');
imwrite(seg, cfg.color_map, 'demo_images/demo_instance_seg.png');