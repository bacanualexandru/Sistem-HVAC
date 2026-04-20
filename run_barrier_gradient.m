% FUNCȚIA GRADIENT
% ====================
function [x_traj, u_traj, all_costs_grad] = run_barrier_gradient(A, B, Q, R, x0, x_ref, u_ref, N, T_sim, u_ub, u_lb, disturbance_traj)
n = size(A,1); m = size(B,2);
x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);
all_costs_grad = cell(1, T_sim);

epsilon = 0.001;
sigma = 0.5;
alpha = 0.01;

for t = 1:T_sim
    [H, q, C, d] = denseMpcBariera(A, B, Q, R, x0, N, u_ub, u_lb, x_ref, u_ref);
    x_barrier = zeros(size(H, 1), 1);
    thau = 1;
    m_constraints = size(C,1);

    gradient = H * x_barrier + q;  % asigurăm inițializarea
    cost_iter = [];

    while m_constraints * thau >= epsilon
        ci = C * x_barrier - d;
        gradient = H * x_barrier + q;
        for i = 1:m_constraints
            gradient = gradient - thau * (C(i,:)') / ci(i);
        end
        k = 0;
        while norm(gradient) > 1e-3 && k < 100
            cost_iter(end+1) = norm(gradient);  % salvăm înainte de update
            x_barrier = x_barrier - alpha * gradient;
            ci = C * x_barrier - d;
            gradient = H * x_barrier + q;
            for i = 1:m_constraints
                gradient = gradient - thau * (C(i,:)') / ci(i);
            end
            k = k + 1;
        end
        thau = thau * sigma;
    end

    all_costs_grad{t} = cost_iter;

    u_applied = x_barrier(1:m);
    x0 = A * x0 + B * u_applied + disturbance_traj(:,t);
    x_traj(:,t+1) = x0;
    u_traj(:,t) = u_applied + disturbance_traj(:,t);
end
end