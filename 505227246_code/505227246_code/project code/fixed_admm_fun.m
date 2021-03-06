function [x,iter] = fixed_admm_fun(y,H,T,Q,K,N,M,SNR,lam,max_iter,tolerance)
%%%%%%%%%%%%%%%%%%% convert to real values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% yr = real(y);
% yi = imag(y);
ytrans = y;
size(ytrans);
% Hr = real(H);
% Hi = imag(H);
Htrans = H;
size(Htrans);
x = zeros(K,1);
lambda = lam;
% c_a = 0.1;
%%%%%%%%%% step -1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
support_set_prev = zeros(2*K,1);
support_set_current = zeros(2*K,1);

iter = 1;

objective_val = [];
next = inf;

alpha_prev = 10*ones(T,1);
alpha_current = 10*ones(T,1);

x_prev = zeros(2*K,T);
x_current = zeros(2*K,T);

x_prev_hat = zeros(2*K,T);
x_current_hat = zeros(2*K,T);

r = Inf(2*K,T);
s = Inf(2*K,T);

z_prev = zeros(2*K,T);
z_current = zeros(2*K,T);

u_prev = zeros(2*K,T);
u_current = zeros(2*K,T);

w_prev = ones(2*K,1);
w_current = ones(2*K,1);

% r_store = zeros(max_iter,1);
% s_store = zeros(max_iter,1);

row = ones(T,1);
    %%%%%%%%%%%%% efficient  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V = zeros(2*K,2*K,T);
    L = zeros(2*K,2*K,T);
    
    for t = 1:T
        [V(:,:,t),L(:,:,t)] = eig(Htrans(:,:,t)'*Htrans(:,:,t));
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (norm(r,2) > tolerance || norm (s,2) > tolerance) && max_iter >= iter
% while max_iter >= iter
    


    
    %%%%%%%%%%%%%%%%%%%% step-3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t = 1:T
        %%%%%%%%%%%%%%% step-4 %%%%%%%%%%%%
        x_current(:,t) = V(:,:,t)*inv(L(:,:,t)+(row(t) + alpha_prev(t,1))*eye(2*K))*V(:,:,t)'*(Htrans(:,:,t)'*ytrans(:,t) + alpha_prev(t,1)*x_prev_hat(:,t) + row(t)*(z_prev(:,t)-u_prev(:,t)));
        z_current(:,t) = max((x_current(:,t) + u_prev(:,t)- (lambda*w_prev(:,1)/row(t))), zeros(2*K,1)) - max((-x_current(:,t) - u_prev(:,t)- (lambda*w_prev(:,1)/row(t))), zeros(2*K,1));
        %%%%%%%%%%%%% step-5 %%%%%%%%%%%%%
        u_current(:,t) = u_prev(:,t) + x_current(:,t)- z_current(:,t);
    end
    %%%%%%%%%%%%%%%%% obj  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     if abs(next - objective_val(iter)) < tolerance
%         break;
%     end

    
    %%%%%%%%%%%%%%%%% step-7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z_total = zeros(2*K,1);
    for t = 1:T
        z_total = z_total + abs(z_current(:,t));
    end
    [z_sorted, I] = sort(abs(z_total));
    z_inf = norm(z_sorted,inf);
%     if z_inf == z_sorted(2*K)
%         disp('yay')
%         iter
%     end
    %%%%%%%%%%%%%% step-8 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ind = 1;
    while z_sorted(ind+1) - z_sorted(ind) <= (7/(2*N))*(z_inf/min(iter,6)) && ind < 2*K-1  
    ind = ind+1;
    end
    beta = abs(z_sorted(ind));
    %%%%%%%%%%%%%%%% step-9 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    support_set_current = zeros(2*K,1);
    for j = 1:2*K
     if abs(z_total(j))>beta
         support_set_current(j) = 1;
     end
    end
    
    w_current = ones(2*K,1);
    for j = 1:2*K
        if support_set_current(j) == 1
            w_current(j) = 0;
        end
    end
    x_current_hat = z_current;
    %%%%%%%%%%%%%%%% step-10 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    r = x_current - z_current;
    s = -(z_current - z_prev).*row';
%     r_store(iter) = norm(r,2);
%     s_store(iter) = norm(s,2);
    
   %%%%%%%%%%%%%%%%%%%%%%%%%  Objective %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%% transformations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ytrans_obj = zeros(2*(N+K),T); 
    
    for t = 1:T
        ytrans_obj(:,t) = [y(:,t);sqrt(alpha_current(t))*x_prev_hat(:,t)];
    end
    
    Htrans_obj = zeros(2*(N+K),2*K,T);
    
    for t = 1:T
        Htrans_obj(:,:,t) = [H(:,:,t) ; sqrt(alpha_current(t))*eye(2*K)];
    end
   
    objective_val(iter) = objective(Htrans_obj,ytrans_obj,lam,x_current,z_current,w_prev,T);
    next = objective_val(iter);

    iter = iter +1;

%%%%%%%%%%%%% updates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     support_set_prev = support_set_current;
    alpha_prev = alpha_current;
%     x_prev = x_current;
    x_prev_hat = x_current_hat;
    z_prev = z_current;
    u_prev = u_current;
    w_prev = w_current;

end
iter = iter-1;
dk = ['Iterations: ',num2str(iter)];
disp(dk)
x = x_current;
obj2 = (objective_val- objective_val(iter))/(objective_val(iter));
it = 1:length(objective_val);
% % figure
% % hold on;
% % plot(it,r_store)
% % plot(it, s_store)
plot(it,obj2,'^')
it = 1:max_iter;
plot(it, r_store,'o')




