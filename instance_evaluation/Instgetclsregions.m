function clsregions=Instgetclsregions(image)
    clsregions=unique(image);               % get labels of regions
    clsregions=clsregions(clsregions>0);    % exclude background
end