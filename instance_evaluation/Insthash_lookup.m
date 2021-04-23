function ind = Insthash_lookup(hash,s, Opts)

hsize=numel(hash.key);
h=mod(Instimgname2num(s, Opts),hsize)+1;
ind=hash.val{h}(strcmp(s,hash.key{h}));
