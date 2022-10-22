clear;clc;
classnum=15;totlenuminclass=11;
load ./datasets/Yale5040165; A=Yale5040165;  
Breshape=A(1:50,1:40,:); 
X=reshapeimageCOLtovector(Breshape);
load ./datasets/randvector10by11
clear Yale5040165
trainingnuminclass= 4; 

for traintime=1:10
    clear  Xtrain_label Xtrain Xtest_label Xtest 
    randvector=randvector10by11(traintime,:);

    ind=0;
    for i=1:classnum
        for j=1:totlenuminclass
            count=(i-1)*totlenuminclass+randvector(j);
            ind=ind+1;
            Xrand(:,ind)=X(:,count); 
        end
    end
    Y = zeros(classnum, classnum * trainingnuminclass);
    count=0;ind=0;
    clear gnd Xtrain_label Xtrain
    for i=1:classnum
        for j=1:trainingnuminclass
            count=(i-1)*totlenuminclass+randvector(j);
            ind=ind+1;
            Xtrain(:,ind)=Xrand(:,count);
            gnd(ind)=i;Xtrain_label(ind)=i; 
             Y( Xtrain_label(ind), ind) = 1;
        end
    end

    count2=0;ind2=0; clear Xforconstructgraph XXrand Xtest_label Xtest
    for i=1:classnum
        for j=1+trainingnuminclass:totlenuminclass
            count2=(i-1)*totlenuminclass+j;
            ind2=ind2+1;Xtest_label(ind2)=i;
            Xtest(:,ind2)=Xrand(:,count2);
        end
    end   
    clear Ytrain Ytest
    PCAdim = 50;
    reducedim = 45; 
    xzhou=[];
    neighK=5;
    [eigvectorPCA] = PCA_dencai(Xtrain', PCAdim);
    Xtrain1=eigvectorPCA'*Xtrain;
    Xtest1=eigvectorPCA'*Xtest;     
    Z=zeros(classnum*trainingnuminclass,classnum);
    k=0;
    for i=1:classnum
        for j=1:trainingnuminclass
            Z(j+k,i) = 1;
        end
        k=k+trainingnuminclass;
    end 
    [Y3,A,B3] = JSER(Xtrain1',Xtrain_label',Z,45,10^1,10^9,10^7,5,80,10,3);
    for dim3=5:5:size(B3,2)
           projection3=B3(:,1:dim3);
           Ytrain3=projection3'*Xtrain1;
           Ytest3=projection3'*Xtest1;    
           rateYale(traintime,dim3/5)= KNN_Classfier(Ytrain3, Xtrain_label, Ytest3,Xtest_label, 1)
           xzhou(dim3/5)=dim3;
    end
        
end
Recog_rate=sum(rateYale,1)./traintime;
[m,index]=max(Recog_rate)
std(Recog_rate)
plot(xzhou,Recog_rate);




