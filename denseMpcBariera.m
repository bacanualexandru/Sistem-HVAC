function [H, q, C, d] = denseMpcBariera(A, B, Q, R, z0, N, u_ub, u_lb, z_ref, u_ref)
    [nz, nu] = size(B); % nz = nr stari, nu = nr comenzi
    % extind Q si R pe orizontul N
    QQ = kron(eye(N), Q);
    RR = kron(eye(N), R);
    % construiesc extensiile matricei A si B (pt predictie pe orizont)
    A_ext = zeros(N*nz, nz);
    B_ext = zeros(N*nz, N*nu);
    for i = 1:N
        A_ext((i-1)*nz+1:i*nz, :) = A^i;
        for j = 1:i
            B_ext((i-1)*nz+1:i*nz, (j-1)*nu+1:j*nu) = A^(i-j) * B;
        end
    end
    zz_ref = kron(ones(N,1), z_ref);  % stari tinta% vectori tinta (replicati pe orizont)
    uu_ref = kron(ones(N,1), u_ref);  % comenzi tinta (zero)
    bb = A_ext * z0;% starea initiala propagata
    Cu = kron(eye(N), [eye(nu); -eye(nu)]);% constrangeri pe comenzi
    du = kron(ones(N,1), [u_ub; -u_lb]);
    H = B_ext' * QQ * B_ext + RR; % cost quadratic: H si q din forma standard QP
    q = B_ext' * QQ * (bb - zz_ref) - RR * uu_ref;
    C = Cu; % constrangeri liniare pe u
    d = du;
end
