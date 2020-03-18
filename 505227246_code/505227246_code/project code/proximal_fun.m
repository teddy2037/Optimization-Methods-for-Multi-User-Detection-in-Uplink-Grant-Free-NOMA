function [x,iter] = proximal_fun(y,H,T,Q,K,N,M,SNR,lam,max_iter,tolerance)
%%%%%%%%%%%%%%%%%%% convert to real values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% yr = real(y);
% yi = imag(y);
% ytrans = y;
% size(ytrans);
% Hr = real(H);
% Hi = imag(H);
Htrans = H;
size(Htrans);

lambda = lam;
c_a = 0.1;

%%%%%%%%%% step -1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

support_set_prev = zeros(2*K,1);
support_set_current = zeros(2*K,1);

iter = 1;

alpha_prev = 10*ones(T,1);
alpha_current = 10*ones(T,1);

x_prev = zeros(2*K,T);
x_current = zeros(2*K,T);

x_prev_hat = zeros(2*K,T);
x_current_hat = zeros(2*K,T);

r = Inf(2*K,T);
s = Inf(2*K,T);

w_prev = ones(2*K,1);
w_current = ones(2*K,1);

grad = zeros(2*K,T);

r_store = zeros(max_iter,1);
s_store = zeros(max_iter,1);

next  = inf;

gamma = 1;
para_beta = 0.4;

eigval = zeros(T,1);

objective_val = [];

row = 1;

    %%%%%%%%%%%%%%5 transformations %%%%%%%%%%%%%%%%%%%%%%%%%%
    ytrans = zeros(2*(N+K),T); 
    
    for t = 1:T
        ytrans(:,t) = [y(:,t);sqrt(alpha_prev(t))*x_prev_hat(:,t)];
    end
    
    Htrans = zeros(2*(N+K),2*K,T);
    
    for t = 1:T
        Htrans(:,:,t) = [H(:,:,t) ; sqrt(alpha_prev(t))*eye(2*K)];
    end
    AtA = zeros(2*K,2*K,T);
    Atb = zeros(2*K,T);
    
    for t = 1:T
      AtA(:,:,t) = Htrans(:,:,t)'*Htrans(:,:,t);
      eigval(t,1) = 1/max(eig(AtA(:,:,t)));
      Atb(:,t) = Htrans(:,:,t)'*ytrans(:,t);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while max_iter >= iter && norm (s,2) > tolerance
    

    AtA = zeros(2*K,2*K,T);
    Atb = zeros(2*K,T);
    
    for t = 1:T
      AtA(:,:,t) = Htrans(:,:,t)'*Htrans(:,:,t);
%       eigval(t,1) = 1/max(eig(AtA(:,:,t)));
      Atb(:,t) = Htrans(:,:,t)'*ytrans(:,t);
    end
    

    grad_x = zeros(2*K,T);
    %%%%%%%%%%%%%%%%%%%% step-3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t = 1:T
        %%%%%%%%%%%%%%% step-4 %%%%%%%%%%%%
        grad_x(:,t) = AtA(:,:,t)*x_prev(:,t) - Atb(:,t); 
        x_current(:,t) = prox_l1(x_prev(:,t) - eigval(t)*grad_x(:,t), eigval(t)*lambda, w_prev);
    end

%     if abs(next - objective_val(iter)) < 0.1*tolerance
%         break;
%     end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update gamma %%%%%%%%%%%%%
%     gamma = para_beta*gamma;
    %%%%%%%%%%%%%%%%% step-7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_total = zeros(2*K,1);
    for t = 1:T
        x_total = x_total + abs(x_current(:,t));
    end
    [x_sorted, I] = sort(abs(x_total));
    x_inf = norm(x_sorted,inf);
    %%%%%%%%%%%%%% step-8 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ind = 1;
    while x_sorted(ind+1) - x_sorted(ind) <= (7/(2*N))*(x_inf/min(iter,6)) && ind < 2*K-1  
    ind = ind+1;
    end
    ind;
    beta = abs(x_sorted(ind));
    %%%%%%%%%%%%%%%% step-9 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    support_set_current = zeros(2*K,1);
    for j = 1:2*K
     if abs(x_total(j))>beta
         support_set_current(j) = 1;
     end
    end
    w_current = ones(2*K,1);
    for j = 1:2*K
        if support_set_current(j) == 1
            w_current(j) = 0;
        end
    end
    w_current;
    x_current_hat = x_current;
    %%%%%%%%%%%%%%%% step-10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    s = -row*(x_current - x_prev);
%     size(r)
%     size(s)
   
    s_store(iter) = norm(s,2);
    %%%%%%%%%%%%%%%%%% step-11 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     for t =1:T
%     alpha_current(t) = (2*K*c_a)/(2*N*(norm(s(:,t),2)));
%     end

ytrans = zeros(2*(N+K),T); 
    
    for t = 1:T
        ytrans(:,t) = [y(:,t);sqrt(alpha_current(t))*x_current_hat(:,t)];
    end
    
    Htrans = zeros(2*(N+K),2*K,T);
    
    for t = 1:T
        Htrans(:,:,t) = [H(:,:,t) ; sqrt(alpha_current(t))*eye(2*K)];
    end    

objective_val(iter) = objective(Htrans,ytrans,lam,x_current,x_current,w_prev,T);
   
iter = iter +1;

%%%%%%%%%%%%% updates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    support_set_prev = support_set_current;
    alpha_prev = alpha_current;
    x_prev = x_current;
    x_prev_hat = x_current_hat;
    w_prev = w_current;

end
iter = iter-1;
dk = ['Iterations: ',num2str(iter)];
disp(dk)
x = x_current;
obj2 = (objective_val- objective_val(iter))/(objective_val(iter));
it = 1:length(objective_val);
plot(it,obj2,'^')
% figure
% hold on;
it = 1:max_iter;
plot(it, s_store,'o')
% plot(it,r_store)







