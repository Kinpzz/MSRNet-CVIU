function num = Instimgname2num(path, Opts)

strs=regexp(path, '_', 'split');
preffix=Opts.preffixmap(strs{1});
str=[num2str(preffix) strs{2}];
num=str2double(str);