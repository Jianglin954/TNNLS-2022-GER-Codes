function [L,Lp,W,Wp] = GetLandLpByWandWp(gnd, options, data)


if (~exist('options','var'))
   options = [];
end


[nSmp,nFea] = size(data);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

intraK = 5;
if isfield(options,'intraK') 
    intraK = options.intraK;
end

interK = 20;
if isfield(options,'interK') 
    interK = options.interK;
end

tmp_T = cputime;

Label = unique(gnd);
nLabel = length(Label);


D = EuDist2(data,[],0);

nIntraPair = 0;
if intraK > 0
    G = zeros(nSmp*(intraK+1),3); 
    idNow = 0;
    for i=1:nLabel
        classIdx = find(gnd==Label(i)); 
        DClass = D(classIdx,classIdx); 
        [dump idx] = sort(DClass,2); 
        clear DClass dump;
        nClassNow = length(classIdx);
        nIntraPair = nIntraPair + nClassNow^2;
        if intraK < nClassNow
            idx = idx(:,1:intraK+1); 
        else
            idx = [idx repmat(idx(:,end),1,intraK+1-nClassNow)];
        end

        nSmpClass = length(classIdx)*(intraK+1); 
        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[intraK+1,1]);
        G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:)); 
        G(idNow+1:nSmpClass+idNow,3) = 1;        
        idNow = idNow+nSmpClass;
        clear idx
    end
    Sc = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
  
    [I,J,V] = find(Sc);
    Sc = sparse(I,J,1,nSmp,nSmp);
    Sc = max(Sc,Sc');
    clear G
else
    Sc = zeros(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        nClassNow = length(classIdx);
        nIntraPair = nIntraPair + nClassNow^2;
        Sc(classIdx,classIdx) = 1;
    end
end


if interK > 0 & (interK < (nSmp^2 - nIntraPair))
    maxD = max(max(D))+100;
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        D(classIdx,classIdx) = maxD;
    end
    
    [dump,idx] = sort(D(:));
    clear dump D
    idx = idx(1:interK);
    [I, J] = ind2sub([nSmp,nSmp],idx);
    Sp = sparse(I,J,1,nSmp,nSmp);
    Sp = max(Sp,Sp');
else
    Sp = ones(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(gnd==Label(i));
        Sp(classIdx,classIdx) = 0;
    end
end

%Lp=Dp-Lp
Dp = full(sum(Sp,2));
Wp = Sp;
Sp = -Sp;
for i=1:size(Sp,1)
    Sp(i,i) = Sp(i,i) + Dp(i);
end
%Lc=Dc-Sc
Dc = full(sum(Sc,2));
W = Sc;
Sc = -Sc;
for i=1:size(Sc,1)
    Sc(i,i) = Sc(i,i) + Dc(i);
end
L =  Sc;
Lp = Sp;

timeW = cputime - tmp_T;
