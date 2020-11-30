function [map_out] = map_refine(map, threshold)
% pruning out some small resions (smaller than threshold)
CC = bwconncomp(map,8);
map_out = map;
for ii=1:CC.NumObjects
    if(length(CC.PixelIdxList{ii})<=threshold)
        map_out(CC.PixelIdxList{ii})=0;
    end
end
return