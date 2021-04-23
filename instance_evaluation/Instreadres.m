function [imgs, regions, ovmaxs, confidence]=Instreadres(gtids, Opts)

MAX=5000;
imgs=cell(MAX,1);            % imgs name
regions=zeros(MAX,1);        % instance id
ovmaxs=zeros(MAX,1);         % instance overmax with gt
confidence=zeros(MAX,1);     % instance confidence

idx=0;
% scores(confidence)
if confidence
    load(sprintf('%s/score.mat',Opts.respath));
end
% label mapping
load(sprintf('%s/label_mapping.mat', Opts.respath2));

for imageid=1:length(gtids)
  if confidence
      if ~strcmp(gtids{imageid},scores(imageid).name)
          error('Not match!');
      end
      res_scores=scores(imageid).scores;
  end
  mapping=label_mapping(imageid).mapping;
  % load image
  [gt, ~]=imread(sprintf('%s/%s%s',Opts.annopath, gtids{imageid},Opts.suffix));
  % read the original predictions and relabeled ones
  res1=imread(sprintf('%s/%s%s',Opts.respath,gtids{imageid},Opts.suffix));
  res2=imread(sprintf('%s/%s%s',Opts.respath2,gtids{imageid},Opts.suffix));

  resregions1 = Instgetclsregions(res1);
  %resregions2 = Instgetclsregions(res2);

  for regionid=1:length(resregions1)

    if mapping(regionid)==0
      ovmax=0; 
    else
      % find the max overlap
      regionid_relabel=mapping(regionid);
   
      gtmask=(gt == regionid_relabel);
      resmask= (res2 == regionid_relabel);
      ovmax=CalIOU(gtmask,resmask);
    end
    idx=idx+1;
    imgs{idx}=gtids{imageid};
    regions(idx)=regionid_relabel;
    ovmaxs(idx)=ovmax;
    if confidence
        if isempty(res_scores)
            res_scores = 1;
        end
        confidence(idx)=res_scores(regionid);
    else
        confidence(idx) = ovmax;
    end
%     if res_scores(regionid) < 0.1
%         fprintf('%.4f %.4f\n', res_scores(regionid),ovmax);
%     end
  end
  
%   for regionid=1:length(resregions2)
%     gtmask=(gt == resregions2(regionid));
%     resmask= (res2 == resregions2(regionid));
%     score=CalIOU(gtmask,resmask);
%     
%     idx=idx+1;
%     imgs{idx}=gtids{imageid};
%     regions(idx)=regionid;
%     ovmaxs(idx)=score;                             % to be  modified  
%   end
end

imgs=imgs(1:idx);
regions=regions(1:idx);
ovmaxs=ovmaxs(1:idx);
confidence=confidence(1:idx);
