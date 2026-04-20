% FUNCȚIA CVX
% ====================
function [x_traj, u_traj] = run_cvx(A, B, Q, R, x0, x_ref, lambda, mu, N, T_sim, disturbance_traj)
n = size(A,1); m = size(B,2);
Qblk = kron(eye(N), Q);
Rblk = kron(eye(N), R);
D = diff(eye(N));
S = mu * kron(D' * D, eye(m));

x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);

for k = 1:T_sim
    M = zeros(N*n, N*m); x_vec = zeros(N*n, 1);
    for i = 1:N
        x_vec((i-1)*n+1:i*n) = A^i * x0;
        for j = 1:i
            M((i-1)*n+1:i*n, (j-1)*m+1:j*m) = A^(i-j) * B;
        end
    end
    H = 2 * (M' * Qblk * M + Rblk + S);
    q = 2 * M' * Qblk * (x_vec - kron(ones(N,1), x_ref));

    C1 = eye(N*m); C2 = -eye(N*m);
    d1 = ones(N*m,1); d2 = ones(N*m,1);
    Cx1 = M; dx1 = 0.7 * ones(N*n,1) - x_vec;

    Cx2 = -M; dx2 = -0.4 * ones(N*n,1) + x_vec;

    C = [C1; C2; Cx1; Cx2]; d = [d1; d2; dx1; dx2];

    cvx_begin quiet
        variables X(n, N+1) U(m, N)
        cost = 0;
        for kk = 1:N
            cost = cost + sum_square(X(:,kk) - x_ref) + lambda * sum_square(U(:,kk));
        end
        for kk = 1:N-1
            cost = cost + mu * sum_square(U(:,kk+1) - U(:,kk));
        end
        minimize(cost)
        subject to
            X(:,1) == x0;
            for kk = 1:N
                X(:,kk+1) == A*X(:,kk) + B*U(:,kk);
                0.4 <= X(:,kk) <= 0.7;
                -1 <= U(:,kk) <= 1;
            end
    cvx_end

    u_applied = U(:,1);
    x0 = A*x0 + B*u_applied + disturbance_traj(:,k);
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied + disturbance_traj(:,k);
end
end