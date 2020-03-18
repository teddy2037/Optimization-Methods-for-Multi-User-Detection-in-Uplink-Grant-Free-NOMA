clear all;                                            
close all;
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trials = 100;
snr_max = 10;
snrdb = 2:1:snr_max;
ber = zeros(length(snrdb),trials);

% System Parameter
T = 10;
Q = 4;
K = 108;  % #. users
N = 72;  % #. subcarriers
M = 20;   % #.active users
% SNR = 6;
max_iter = 100;


for snrdb=2:1:snr_max
    
    ber_total = zeros(length(snrdb),1);
    for in = 1:1:trials
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
    xenc;
    
  
  % y
    y0=zeros(2*N,T);
    y=zeros(2*N,T);
    for t=1:T
        y0(:,t) = Htrans(:,:,t)*xenc(:,t);
        y(:,t) = awgn(y0(:,t), snrdb, 'measured');
       
    end
    y;
    
    lam = 0.01;
    
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%% Proximal Methods %%%%%%%%%%%%
    x_recv = proximal_fun(y,Htrans,T,Q,K,N,M,snrdb,lam,max_iter);
  
    %%%%%%%%%%%%%%%%%%% decode algorithm %%%%%%%%%%%%%%%%%
    %%%%%%%% Pick the 12 most significant users %%%%%%%%%%
    xmod = zeros(2*K,T);
    xmod = sum(abs(x_recv),2);
    [xsort, Index_sort] = sort(xmod, 'descend');
    decode_index = Index_sort(1:2*M);
    xdec = zeros(2*K,T);
    for t = 1:T
        for j = 1:2*M
            if x_recv(decode_index(j),t)>=0 
                xdec(decode_index(j),t) = 1;
            else
                xdec(decode_index(j),t) = -1;
            end
        end
    end
%     
%     xdec;
%     xenc;
%  numerrs_psk = zeros(T,1);
%  for t = 1:T
%  numerrs_psk(t,1) = symerr(xenc(:,t),xdec(:,t));
%  end
%  ber = sum(numerrs_psk)/(2*K*T)
%     
 
%%%%%%%%%%%%%%%%%%%%%%%%%%% ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     x_recv = admm_fun(y,Htrans,T,Q,K,N,M,snrdb,lam,max_iter);
%   
%     %%%%%%%%%%%%%%%%%%% decode algorithm %%%%%%%%%%%%%%%%%
%     %%%%%%%% Pick the 12 most significant users %%%%%%%%%%
%     xmod = zeros(2*K,T);
%     xmod = sum(abs(x_recv),2);
%     [xsort, Index_sort] = sort(xmod, 'descend');
%     decode_index = Index_sort(1:2*M);
%     xdec = zeros(2*K,T);
%     for t = 1:T
%         for j = 1:2*M
%             if x_recv(decode_index(j),t)>=0 
%                 xdec(decode_index(j),t) = 1;
%             else
%                 xdec(decode_index(j),t) = -1;
%             end
%         end
%     end
    
%     xdec;
%     xenc;
%  numerrs_psk = zeros(T,1);
%  for t = 1:T
%  numerrs_psk(t,1) = symerr(xenc(:,t),xdec(:,t));
%  end
%  ber = sum(numerrs_psk)/(2*K*T)
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
numerrs_psk = zeros(T,1);
for t = 1:T
 numerrs_psk(t,1) = symerr(xenc(:,t),xdec(:,t));
end
ber(snrdb-1,in) = sum(numerrs_psk)/(2*K*T);
    end 
    
end

for snrdb = 2:1:snr_max
    ber_total(snrdb-1) = sum(ber(snrdb-1,:))/trials;
end

snrdb = 2:1:snr_max;
semilogy(snrdb, ber_total,'o-')
    
