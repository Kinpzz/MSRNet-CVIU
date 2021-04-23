% ------------------------------------------------------------------------ 
%  Copyright (C)
%  HCP, Sun Yat-sen University 2018. All rights reserved.
% 
%  Yuan Xie <xiey39@mail2.sysu.edu.cn>
%
%  Find the best correspondence between each predicted instance and 
%  ground-truth instance according to pixel overlap counts (IoU)
% ------------------------------------------------------------------------ 
function InstRelabel(Opts)
    % load ground truth
    [ids]=textread(sprintf('%s/%s.txt',Opts.imgsetpath,Opts.set),'%s');

    if ~exist(Opts.respath2,'dir')
        mkdir(Opts.respath2);
    end
    %if length(dir(sprintf('%s/*%s',Opts.respath2,Opts.suffix)))==length(ids)
    %    return;
    %end
    for i =1:length(ids)
      id=ids{i};
      [gt, color_map]=imread(sprintf('%s/%s%s',Opts.annopath, id,Opts.suffix));
      [res,~]=imread(sprintf('%s/%s%s',Opts.respath,id,Opts.suffix));
      res = imresize(res, size(gt), 'nearest');
      % extract salient instance
      gt_indices=unique(gt);
      res_indices=unique(res);   
      gt_indices=gt_indices(gt_indices>0);     % exclude background
      res_indices=res_indices(res_indices>0);  % exclude background
      gt_num=length(gt_indices);
      res_num=length(res_indices);
      
      % Calculate IoU scores between each gt instance and pre instance
      % store the scores in scores_map: [N * M]
      %   N: number of instances in groundtruth
      %   M: number or instances in predictions
      scores_map=zeros(gt_num, res_num);
      for gt_idx=1:gt_num
          gtmask=(gt==gt_indices(gt_idx));
        for res_idx=1:res_num
          resmask=(res==res_indices(res_idx));
          score=CalIOU(gtmask,resmask);
          scores_map(gt_idx, res_idx)=score;
        end
      end
      

      res_relabel=zeros(res_num,1);
      while numel(find(scores_map))>0 && numel(find(res_relabel))<res_num
        [ovmax, index]=max(scores_map(:));
        res_idx=ceil(index/gt_num);
        gt_idx=index - (res_idx-1)*gt_num;
        
        if res_relabel(res_idx)==0
            res_relabel(res_idx)=gt_idx;
        end
        scores_map(gt_idx,:)=0;
      end
      % store the mapping
      label_mapping(i)=struct('name',id, 'mapping', res_relabel);
      % construct a relabel result
      result=zeros(size(res));
      temp_idx = 1;
      for idx=1:res_num
          if(res_relabel(idx))>0
            result(res==res_indices(idx))=res_relabel(idx);
          else
            result(res==res_indices(idx))=max(res_relabel) + temp_idx;
            temp_idx = temp_idx + 1;
          end
      end
      %imshow(uint8(result),color_map);
      %path=sprintf('%s_pre/%s%s',Opts.respath, id,Opts.suffix);
      %imwrite(res,color_map,path);
      %path=sprintf('%s_gt/%s%s',Opts.respath, id,Opts.suffix);
      %imwrite(gt,color_map,path);
      
      % save the relabeled map
      path=sprintf('%s/%s%s',Opts.respath2, id,Opts.suffix);
      imwrite(uint8(result),color_map,path);
    end
    savepath=sprintf('%s/label_mapping.mat', Opts.respath2);
    save(savepath,'label_mapping');
   
end