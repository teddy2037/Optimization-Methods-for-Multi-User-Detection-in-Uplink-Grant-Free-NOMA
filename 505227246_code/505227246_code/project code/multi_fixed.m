clear all;                                            
close all;
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System Parameter
T = 10;
Q = 4;
K = 108; 
N = 72; 
M = 20;   
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
  
    
    lam = 0.1;
    tolerance = 0.01;
figure
hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
disp('fista_fun')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_recv = fista_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
toc
xdec = decode_main(x_recv,T,M,K); 
ber = ber_main(xenc,xdec,T,K);  
d = ['ber: ',num2str(ber)];
disp(d)
disp(' ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
disp('fixed_mod_admm_fun')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_recv = fixed_mod_admm_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
toc
xdec = decode_main(x_recv,T,M,K); 
ber = ber_main(xenc,xdec,T,K);  
d = ['ber: ',num2str(ber)];
disp(d)
disp(' ')    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
disp('fixed_fast_admm_fun')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_recv = fixed_f_admm_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
toc
xdec = decode_main(x_recv,T,M,K); 
ber = ber_main(xenc,xdec,T,K);  
d = ['ber: ',num2str(ber)];
disp(d)
disp(' ')    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
disp('proximal_fun')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_recv = proximal_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
toc
xdec = decode_main(x_recv,T,M,K); 
ber = ber_main(xenc,xdec,T,K);  
d = ['ber: ',num2str(ber)];
disp(d)
disp(' ') 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% display method %%%%%%%%%%%%%%%%%%%%%%%%%%
disp('nesterov')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_recv = fast_proximal_fun(y,Htrans,T,Q,K,N,M,SNR,lam,max_iter,tolerance);
toc
xdec = decode_main(x_recv,T,M,K); 
ber = ber_main(xenc,xdec,T,K);  
d = ['ber: ',num2str(ber)];
disp(d)
disp(' ') 
hold off;

    
