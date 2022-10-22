function [B]=reshapeimageCOLtovector(allimagematrix)
%colunm into colunm

[mline,mcolumn,mthick]=size(allimagematrix); %mline * ncolumn * ksample
lenth=mline*mcolumn;
B=zeros(lenth,mthick);
for t=1:mthick
    B(:,t)=reshape(allimagematrix(:,:,t),lenth,1);    %(mline*ncolumn) * ksample ,then 1 column is a sample
end