clear all;                                            
close all;
clc


scale = 1:0.5:3;
points = length(scale);
time_iter_admm = zeros(points,1);
%%%%%%%%%%% mod %%%%%%%%%%%%%%%%%%%%%%
time_iter_fadmm = zeros(points,1);
%%%%%%%%%%% mod %%%%%%%%%%%%%%%%%%%%%%
time_iter_sadmm = zeros(points,1);

for p = 1:points
 scale(p)
 lam = 0.1;
 iter_count = 100;
 
 %%%%%%%% admm %%%%%%%%%%
 time_admm = 0;
 %%%%%%% mod %%%%%%%%%%%%
 time_fadmm = 0;
  %%%%%%% fast %%%%%%%%%%%%
 time_sadmm = 0;
for it = 1:iter_count
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System Parameter
T = 10;
Q = 4;
K = 108*scale(p); 
N = 72*scale(p); 
M = 20*scale(p);
SNR = 5;
max_iter = 400;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Generate Phi
    load pn_all.mat pn420;
    PN_len = K;
    PN255 = transpose(pn420(1,420-PN_len+1:420));
    PN255 = sqrt(PN_len)*ifft(PN255);
    PN255_matrix = zeros(PN_len, PN_len);
    for i = 1:PN_len
        PN255_matrix(i:PN_len, i) = PN255(1:PN_len-i+1);
    end
    Phi_l = zeros(2*PN_len, PN_len);
    for i = 1:PN_len
        Phi_l(i+PN_len:2*PN_len, i) = PN255(1:PN_len-i+1);
        Phi_l(i:i+PN_len-1,i) = PN255(1:PN_len);
    end
    Phi = Phi_l(K*2-N+1:K*2, :);
    for i = 1:K
        Phi(:,i) = Phi(:,i)/norm(Phi(:,i));
    end
    Phi;
    % Generate H
    H=zeros(N,K,T);
    for t=1:T
        h0=random('Normal', 0, 1, N, K);
        H(:,:,t)=h0.*Phi;
    end
    Hr = real(H);
    Hi = imag(H);
    Htrans = [Hr -Hi ; Hi Hr];
    Htrans;
    
    xenc = zeros(2*K,T);
    active_index = randperm(K,M);
    active_index_full = [active_index K+active_index];
    for t = 1:T
        for j = 1:2*M
            xenc(active_index_full(j),t) = 2*randi([0 1],1,1)-1;
        end
        
    end
  
  % y
    y0=zeros(2*N,T);
    y=zeros(2*N,T);
    for t=1:T
        y0(:,t) = Htrans(:,:,t)*xenc(:,t);
        y(:,t) = awgn(y0(:,t), SNR, 'measured');
       
    end
  
    
   
    tolerance = 0.001;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_recv,iter_c] = admm_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
time_admm = time_admm+toc;
% xdec = decode_main(x_recv,T,M,K); 
% ber_admm = ber_main(xenc,xdec,T,K);  
% ber_count_admm = ber_count_admm + ber_admm;
% iteration_c_admm = iteration_c_admm + iter_c;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_recv,iter_c] = mod_admm_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
time_fadmm = time_fadmm+toc;
% xdec = decode_main(x_recv,T,M,K); 
% ber_fadmm = ber_main(xenc,xdec,T,K);  
% ber_count_fadmm = ber_count_fadmm + ber_fadmm;
% iteration_c_fadmm = iteration_c_fadmm + iter_c;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_recv,iter_c] = f_admm_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
time_sadmm = time_sadmm+toc;
% xdec = decode_main(x_recv,T,M,K); 
% ber_sadmm = ber_main(xenc,xdec,T,K);  
% ber_count_sadmm = ber_count_sadmm + ber_sadmm;
% iteration_c_sadmm = iteration_c_sadmm + iter_c;
end
%%%%%%%%%%%%%% admm %%%%%%%%%%%%%%%%%%%%%%%%%%
% ber_snr_admm(p) = ber_count_admm/iter_count;
% total_iter_admm(p) = iteration_c_admm/iter_count;
time_iter_admm(p) = time_admm/iter_count;
%%%%%%%%%%%%%% mod_admm %%%%%%%%%%%%%%%%%%%%%%%%
% ber_snr_fadmm(p) = ber_count_fadmm/iter_count;
% total_iter_fadmm(p) = iteration_c_fadmm/iter_count;
time_iter_fadmm(p) = time_fadmm/iter_count;
%%%%%%%%%%%%%% fast_admm %%%%%%%%%%%%%%%%%%%%%%%%
% ber_snr_sadmm(p) = ber_count_sadmm/iter_count;
% total_iter_sadmm(p) = iteration_c_sadmm/iter_count;
time_iter_sadmm(p) = time_sadmm/iter_count;
end
%%%%%%%%%%% we have the fucking data %%%%%%%%%%%%%%%%%
% figure
% hold on;
% plot(snr,ber_snr_admm,'^')
% plot(snr,ber_snr_fadmm,'s')
% plot(snr,ber_snr_sadmm,'o')
% hold off;
% 
% figure
% hold on;
% plot(snr,total_iter_admm,'^')
% plot(snr,total_iter_fadmm,'s')
% plot(snr,total_iter_sadmm,'o')

figure
hold on;
scale = 1:0.5:3;
plot(scale,time_iter_admm,'^')
plot(scale,time_iter_fadmm,'s')
plot(scale,time_iter_sadmm,'o')
    
