clear Opts
% settings
DATASET_PATH=''; % path to ILSO dastset
TEST_SET='test_id_1000'; % ILSO-1K: test_id_1000; ILSO-2K: test_id_2000
RESULT_PATH=''; % path to instance segmentation


Opts.respath=RESULT_PATH;
Opts.respath2=sprintf('%s_relabel',Opts.respath);
Opts.annopath=fullfile(DATASET_PATH, 'gt');
Opts.suffix='.png';
Opts.imgsetpath=fullfile(DATASET_PATH, 'ImageSets');
Opts.set=TEST_SET;
Opts.minoverlap=[0.5,0.6,0.7,0.8,0.9];
if strcmp(TEST_SET, 'test_id_1000')
    Opts.preffixmap=containers.Map({'PASCALContour','MSO','hku','ECSSD','DUT-OMRON'}, ...
                                {1,2,3,4,5});
else
    Opts.preffixmap=containers.Map({'PASCALContour','MSO','hku','ECSSD','DUT-OMRON','ADD'}, ...
                                {1,2,3,4,5,6});
end
Opts.confidence=false; % if true using your own confidence, else using iou as confidence 