% ------------------------------------------------------------------------ 
%  Copyright (C)
%  HCP, Sun Yat-sen University 2018. All rights reserved.
% 
%  Yuan Xie <xiey39@mail2.sysu.edu.cn>
% ------------------------------------------------------------------------ 
clear;

InstOpts;                                       % Evaluation settings
InstRelabel(Opts);                              
draw = true;


fprintf('===================================================\n')
fprintf('Eval Salient Instance Segmentation\n')
%for k=1:length(aps)
%    fprintf('  AP (threhold: %.2f): %.3f\n',Opts.minoverlap(k),aps(k));
%end
[recs,precs,aps] = InstEvaldet(Opts,draw);
fprintf('===================================================\n')
