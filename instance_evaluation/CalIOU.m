function score=CalIOU(mask1, mask2)
    sz1=size(mask1); sz2=size(mask2);
    if any(sz1~=sz2)
        error('The images pair do not have same size,: ( %d x %d), (%d x %d.)', ...
               sz1(1),sz1(2),sz2(1),sz2(2));
    end
    mask1=double(mask1(:));
    mask2=double(mask2(:));
    inter=(mask1'* mask2);
    union=sum(mask1)+sum(mask2)-inter;

    score=sum(sum(inter)) / max(sum(sum(union)), eps);
end

