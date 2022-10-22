function [Y,A,B] = JSER(X,X_label,Z,c,alpha,beta,lamda,intraK,interK,maxStep,maxStepY)

[n,d] = size(X); %X=n*d each row is a sample
[~,k] = size(Z);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Á¨?Ê≠?ÂàùÂßãÂå?                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=randi([1 c],n,c);
B = zeros(d,c);
A = zeros(k,c);  % k classes
U0 = eye(d,d); % U0 = 1/(norm(B(i,:),2))
U1 = eye(n,n);
U2 = eye(n,n);

Y_1 = zeros(n,(n-c));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Á¨?Ê≠?ÊûÑÂª∫L Lp                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = [];
options.intraK = intraK;
options.interK = interK;
options.Regu = 1;
[L,Lp,~,~] = GetLandLpByWandWp(X_label, options, X);
if issparse(L)
    L = full(L);
end
if issparse(Lp)
    Lp = full(Lp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Á¨?Ê≠?ËÆ°ÁÆóK E P Q                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ËÆ°ÁÆóK E
[K,E,~] = svd(Lp);
if issparse(E)
    E = full(E);
end
E_half = E.^.5;

%ËÆ°ÁÆóP Q
ZZZ = inv(K*E_half) * (L+alpha.*U1) * inv(E_half*K');
[P,Q,~] = svd(ZZZ);
if issparse(Q)
    Q = full(Q);
end
Q_half = Q.^.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Á¨?Ê≠?Ëø≠‰ª£ËÆ°ÁÆóA B Y                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

step=0;
stepY=0;
converged=false;
convergedY=false;
while ~converged && step<maxStep
    step=step+1;
    
    %update B  
%     B = alpha .* inv(alpha .* X' * U1 * X + beta .* U0) * X' * U1' * Y ;
    B = inv(X' *(alpha.*U1+lamda.*U2)*X+beta.*U0) * X' *(alpha.*U1*Y + lamda.*U2*Z*A);
    %update U0
    for i=1:d
        U0(i,i) = 1 / (2 * norm(B(i,:),2) + 0.001) ;
    end
    
    %update A
    Asvd = 2 .* lamda .* Z' * U2 * X * B ; 
    [AsvdU, ~, AsvdV] = mySVD(Asvd);
    A = AsvdU * AsvdV';  
    
    %update U2
    for i=1:n
        ZXBA = Z - X * B * A';
        U2(i,i) = 1 / (2 * norm(ZXBA(i,:),2) + 0.001) ;
    end
    
    
    %update Y
    Y_ = alpha .* inv((Q_half*P')') * inv(K*E_half) * U1 * X * B ;
    X_ = Q_half * P' ;

    while ~convergedY && stepY<maxStepY
        stepY=stepY+1;         
        Yz = [Y_,Y_1];
        %update M_
        M_value = X_' * Yz;
        [U3, ~, V3] = mySVD(M_value);
        M_ = U3 * V3' ;

        %update Y_1
        M1=M_(:,(c+1):n); 
        Y_1 = X_ * M1;
    end
    stepY = 0;
    
    M = M_(:,1:c);
    Y = inv(E_half*K') * M ;
    
    %update U1
    for i=1:n
        YXB = Y - X * B;
        U1(i,i) = 1 / (2 * norm(YXB(i,:),2) + 0.001) ;
    end
end

%step4 normalize and return B
nsmpB=size(B);
nsmpA=size(A);
for i=1:nsmpA
   A(i,:)=A(i,:)/norm(A(i,:),2);
end
for i=1:nsmpB
   B(i,:)=B(i,:)/norm(B(i,:),2);
end


