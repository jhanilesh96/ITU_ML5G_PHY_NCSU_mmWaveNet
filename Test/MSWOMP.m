function [H_est index_set]= MSWOMP(yw,Upsilon_w,Psi,K,M,Lr,Nr,Nt,sp)


sz=size(Upsilon_w);
x=zeros(sz(2),1);
residual=yw;

index_set=[];
iter=0;

while(iter<=sp)
   iter=iter+1;
   c = Upsilon_w'*residual;
   c1=sum(abs(c),2);
   [m1,idx]=max(c1);
   index_set=[index_set idx]; %Update support
   
   x= pinv(Upsilon_w(:,index_set))*yw; %Project input signal by WLS
   
   residual=yw-Upsilon_w(:,index_set)*x; %Update residual
     
end

h=Psi(:,index_set)*x;
H_est= reshape(h,[Nr,Nt,K]);
    
end


