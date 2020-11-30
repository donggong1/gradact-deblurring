function ext_map = extend_map(map, s, dirc)
% extend a map with s and direction parameters (dirc)
% s rasidul of expending
if(s(1)<=0 || s(2)<=0)
    ext_map = map;
    return;
end
s = round(s);
ss = s*2+1;
% ss = s;

if(dirc==1)
    % horz
    k = ones(1,ss(2));
elseif(dirc==2)
    % vert
    k = ones(ss(1),1);
else
    % two direction
    k = ones(ss);
end
tmpmap = conv2(map, k, 'same');
tmpmap(tmpmap>0.001) = 1;
ext_map = tmpmap;
return
