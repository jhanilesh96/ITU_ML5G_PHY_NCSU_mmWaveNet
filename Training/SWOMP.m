function [H_est index_set]= SWOMP(yw,Upsilon_w,Psi,K,M,Lr,Nr,Nt,epsilon)


sz=size(Upsilon_w);
x=zeros(sz(2),1);
residual=yw;
MSE= trace(residual'*residual)/(K*M*Lr);

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

h=Psi(:,index_set)*x;
H_est= reshape(h,[Nr,Nt,K]);
    
end


