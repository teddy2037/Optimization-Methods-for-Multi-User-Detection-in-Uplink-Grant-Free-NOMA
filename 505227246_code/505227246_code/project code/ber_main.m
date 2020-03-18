function ber = ber_main(xenc,xdec,T,K)   
numerrs_psk = zeros(T,1);
 for t = 1:T
 numerrs_psk(t,1) = symerr(xenc(:,t),xdec(:,t));
 end
 ber = sum(numerrs_psk)/(2*K*T);