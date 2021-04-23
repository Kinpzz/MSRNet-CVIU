% ------------------------------------------------------------------------ 
%  Copyright (C)
%  HCP, Sun Yat-sen University 2018. All rights reserved.
% 
%  Yuan Xie <xiey39@mail2.sysu.edu.cn>
%
%  Evaluates Salienct Instance Segmentation
%  Calculate AP(r) over given threhold
%  Reference:
%   [1] The PASCAL Visual Object Classes Challenge (VOC2012).
%       http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/index.html
%   [2] Yang, Y., Hallman, S., Ramanan, D., Fowlkes, C.C.: 
%       Layered object models for image segmentation. TPAMI 34(9) (2012)
% ------------------------------------------------------------------------ 

function [recs,precs,aps] = InstEvaldet(Opts,draw)

% load eval set
[gtids]=textread(sprintf('%s/%s.txt',Opts.imgsetpath,Opts.set),'%s');
npos=0;
% extract structured information
% gt(i): 'region': list of region idx, eg, [1,2,3]
%        'det'   : list of matched flags, eg, [0,0,0]
gt(length(gtids))=struct('region',[],'det',[]);

for i=1:length(gtids)

    %load annotation
    gtimg=imread(sprintf('%s/%s%s',Opts.annopath,gtids{i},Opts.suffix));
    gt(i).region = Instgetclsregions(gtimg);
    
    nregion=length(gt(i).region);               % numbers of instance
    gt(i).det=false(nregion,1);
    npos=npos+sum(nregion);    
end

% hash image ids
hash=Insthash_init(gtids, Opts);
        
% load predictions
[ids, regions, ovmaxs, scores]=Instreadres(gtids, Opts);

% sort detections by decreasing scores(confidence)
[sc,si]=sort(-scores);
ids=ids(si);
regions=regions(si);
ovmaxs=ovmaxs(si);
scores=scores(si);


nd=length(scores);
nthres=length(Opts.minoverlap);
recs=zeros(nthres,nd);
precs=zeros(nthres,nd);
aps=zeros(nthres);

fprintf('Number of instances in gt and predictions: %d, %d\n', ...
            npos,length(ids));

for k=1:nthres
    minoverlap=Opts.minoverlap(k);
    % reset
    for g=1:length(gt)
      gt(g).det=zeros(size(gt(g).det));
    end
    tp=zeros(nd,1);
    fp=zeros(nd,1);    
    for d=1:nd
        % retrieve the dth detection
        region=regions(d);
        ovmax=ovmaxs(d);

        % find the corresponding ground truth image
        i=Insthash_lookup(hash,ids{d}, Opts);
        if isempty(i)
            error('unrecognized image "%s"',ids{d});
        elseif length(i)>1
            error('multiple image "%s"',ids{d});
        end

        % assign detection as true positive/false positive 
        % according to given threhold(minoverlap)
        if ovmax>minoverlap
            if ~gt(i).det(region)
                tp(d)=1;                % true positive
                gt(i).det(region)=true;
            else 
                fp(d)=1;                % flase posiitve(multiple detection)
            end
        else
            fp(d)=1;                    % false positive
        end    
    end
    % disp
    
    % compute precision/recall
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/npos;
    prec=tp./(fp+tp);

    ap=VOCap(rec,prec);
    % disp
    fprintf('- Minoverlap: %.2f\n', minoverlap);
    fprintf('-   AP: %.3f (TP: %d, FP: %d)\n', ap, max(tp), max(fp));
    
    recs(k,:)=rec;
    precs(k,:)=prec;
    aps(k)=ap;
    if draw
        % plot precision/recall
        subplot(nthres,1,k);      
        plot(rec,prec,'-');
        grid;
        xlabel 'recall'
        ylabel 'precision'
        title(sprintf('Minoverlap: %.2f, AP = %.3f',minoverlap,ap));
    end

end

