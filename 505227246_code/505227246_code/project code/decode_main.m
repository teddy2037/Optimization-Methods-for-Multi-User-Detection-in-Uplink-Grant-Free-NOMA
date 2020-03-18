function xdec = decode_main(x_recv,T,M,K)   
xmod = sum(x_recv.^2,2);
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