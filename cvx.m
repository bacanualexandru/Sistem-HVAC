clear;clc;
A = [0.9 0.1; 0.1 0.9];% sistemul dinamic
B = [0.2 0.1; 0.1 0.2];
x_ref = [0.6; 0.5];      % umiditate tinta
N = 30;                 % orizont MPC
lambda = 0.1;           % penalizare pe u
mu = 0.3;               % penalizare pe delta u
T_sim = 17;             % cate iteratii simulez
x0 = [0.45; 0.4];        % stare initiala
n = size(A, 1); 
m = size(B, 2); 
Q = kron(eye(N), eye(n));
R = kron(eye(N), lambda * eye(m));
D = diff(eye(N));% pt variatii u
S = mu * kron(D' * D, eye(m));
x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);
% zgomot/perturbatii adaugate
rng(1); 
disturbance_traj = [
    0.001  0.001  0.002  -0.03  0.002  0.002  0.002  0.001  0.003  0.002  ...
    0.001  0.001  0.002  0.002  0.01  0.001  0.002  0.002  0.001  0.001 0;
    0.000  0.001  0.002  0.002  0.001  0.001  0.001  0.000  0.001  0.002  ...
    0.002  0.001  0.001  0.001  0.002  0.002  0.001  0.001  0.002  0.001 0
];
% bucla principala MPC
for k = 1:T_sim
    % construiesc x_vec si M pentru predictie
    M = zeros(N * n, N * m);
    x_vec = zeros(N * n, 1);
    for i = 1:N
        A_power = A^i;
        x_vec((i-1)*n+1:i*n) = A_power * x0;
        for j = 1:i
            M_block = A^(i-j) * B;
            M((i-1)*n+1:i*n, (j-1)*m+1:j*m) = M_block;
        end
    end
    % QP: H si q
    H = 2 * (M' * Q * M + R + S);
    q = 2 * M' * Q * (x_vec - repmat(x_ref, N, 1));
    % constrangeri pe u si x
    C1 = eye(N * m);
    C2 = -eye(N * m);
    d1 = ones(N * m, 1);
    d2 = ones(N * m, 1);
    Cx1 = M;
    dx1 = repmat(0.7, N * n, 1) - x_vec;
    Cx2 = -M;
    dx2 = -repmat(0.4, N * n, 1) + x_vec;
    C = [C1; C2; Cx1; Cx2];
    d = [d1; d2; dx1; dx2];
    % solve cu CVX
    cvx_begin
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
            X(:,kk+1) == A * X(:,kk) + B * U(:,kk);
            0.4 <= X(:,kk) <= 0.7;
            -1 <= U(:,kk) <= 1;
        end
    cvx_end
    % update stare si salvez
    u_applied = U(:,1);
    x0 = A * x0 + B * u_applied+disturbance_traj(:,k);
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied+disturbance_traj(:,k);
end
t = 0:T_sim;
% plot stare
figure;
plot(t, x_traj(1,:), '-', 'LineWidth', 2.2, 'Color', [0.2 0.6 0.9]); hold on;
plot(t, x_traj(2,:), '-', 'LineWidth', 2.2, 'Color', [0.9 0.4 0.2]);
yline(0.6, '--', 'LineWidth', 1.5, 'Color', [0 0.2 0]);        % Ref sera 1
yline(0.5, '--', 'LineWidth', 1.5, 'Color', [0.2 0 0.6]);      % Ref sera 2
yline(0.4, '-', 'LineWidth', 1, 'Color', [0 0 0]);
yline(0.65, '-', 'LineWidth', 1, 'Color', [0 0 0]);
xlabel('Timp [pas]');
ylabel('Umiditate (%)');
title('Evoluția umidității în cele două sere');
legend('Sera 1', 'Sera 2', 'Ref. Sera 1', 'Ref. Sera 2', 'Limită minimă', 'Limită maximă', ...
       'Location', 'best');
grid on;
ylim([0.3 0.8]);
set(gca, 'FontSize', 12);
% plot comenzi
figure;
plot(0:T_sim-1, u_traj(1,:), '-', 'LineWidth', 2.2, 'Color', [0.2 0.6 0.9]); hold on;
plot(0:T_sim-1, u_traj(2,:), '-', 'LineWidth', 2.2, 'Color', [0.9 0.4 0.2]);
yline(0, '--k', 'LineWidth', 1.2);
xlabel('Timp [pas]');
ylabel('Comandă u_t');
title('Comenzi aplicate celor două umidificatoare');
legend('u_1 (Sera 1)', 'u_2 (Sera 2)', 'Zero', 'Location', 'best');
grid on;
ylim([-0.4 0.6]); 
set(gca, 'FontSize', 12);