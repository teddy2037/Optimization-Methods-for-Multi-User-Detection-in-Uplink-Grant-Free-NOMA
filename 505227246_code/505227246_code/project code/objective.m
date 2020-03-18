function p = objective(A, b, lam, x, z, w_prev,T)
    w = diag(w_prev);
    val = zeros(T,1);
    for t = 1:T
     val(t) = 0.5*sum_square(A(:,:,t)*x(:,t) - b(:,t)) + lam*norm(w*z(:,t),1);
    end
    
    p = sum(val)/T;
end