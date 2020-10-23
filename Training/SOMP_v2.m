function [H_est index_set]= SOMP_v2(yw,Upsilon_w,Psi,K,M,Lr,Nr,Nt,epsilon)
% K=Nfft; M=Ntrain;
% epsilon=1e-2;
% epsilon=0.99*var_n;

sz=size(Upsilon_w);
x=zeros(sz(2),1);
%OMP
residual=yw;
MSE= trace(residual'*residual)/(K*M*Lr);

% H_est=MSE;
index_set=[];
iter=0;
while(MSE>epsilon)
   iter = iter +1;
   c = Upsilon_w'*residual;
   c1=sum(abs(c),2);
   [m1,idx]=max(c1);
   index_set=[index_set idx]; %Update support
   
   x= pinv(Upsilon_w(:,index_set))*yw; %Project input signal by WLS
   
   residual=yw-Upsilon_w(:,index_set)*x; %Update residual
   
   MSE= trace(residual'*residual)/(K*M*Lr); %MSE update
  
end
% p=length(index_set);*
% tempx=zeros()
% for j=1:lenght(index_set)

h=Psi(:,index_set)*x;
H_est= reshape(h,[Nr,Nt,K]);
    
end


