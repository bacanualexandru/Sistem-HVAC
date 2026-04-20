clear; clc;

% Parametri model
A = [0.9 0.1;
     0.1 0.9];
B = [0.2 0.1;
     0.1 0.2];

x_ref = [0.6; 0.5];
u_ref = [0; 0];

N = 30;
lambda = 0.1;
mu = 0.3;

T_sim = 17;
x0 = [0.45; 0.4];

n = size(A,1);
m = size(B,2);

Q = eye(n);
R = lambda * eye(m);

x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);

u_ub = 1 * ones(m, 1);
u_lb = -1 * ones(m, 1);

rng(1);
disturbance_traj = [
    0.001  0.001  0.002  -0.03  0.002  0.002  0.002  0.001  0.003  0.002  ...
    0.001  0.001  0.002  0.002  0.01  0.001  0.002;
    0.000  0.001  0.002  0.002  0.001  0.001  0.001  0.000  0.001  0.002  ...
    0.002  0.001  0.001  0.001  0.002  0.002  0.001
];

maxIter = 10;
epsilon = 0.001;
sigma = 0.5;
alpha = 0.01;
for k = 1:T_sim
    % Construim QP: H, q, C, d
    [H, q, C, d] = denseMpcBariera(A, B, Q, R, x0, N, u_ub, ...
        u_lb, x_ref, u_ref);
    x_barrier = zeros(size(H, 1), 1);
    thau = 1;
    m_constraints = size(C, 1);
    steps = 0;
    % Metoda barierei logaritmice + gradient descent
    while m_constraints * thau >= epsilon
        inner_iter = 0;
        while inner_iter < 100
            ci = C * x_barrier - d;
            gradient = H * x_barrier + q;
            for i = 1:m_constraints
                gradient = gradient - thau * (C(i,:)' / ci(i));
            end
            if norm(gradient) < 1e-3
                break;
            end
            x_barrier = x_barrier - alpha * gradient;
            inner_iter = inner_iter + 1;
        end

    thau = thau * sigma;
end
    u_applied = x_barrier(1:m); % Aplică controlul și actualizează starea
    x0 = A * x0 + B * u_applied + disturbance_traj(:,k);
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied + disturbance_traj(:,k);
end
% Afișare grafice
t = 0:T_sim;

figure;
plot(t, x_traj(1,:), '-', 'LineWidth', 2.2); hold on;
plot(t, x_traj(2,:), '-', 'LineWidth', 2.2);
yline(0.6, '--', 'LineWidth', 1.5);
yline(0.5, '--', 'LineWidth', 1.5);
yline(0.4, '-', 'LineWidth', 1);
yline(0.7, '-', 'LineWidth', 1);
xlabel('Timp [pas]');
ylabel('Umiditate');
legend('Sera 1','Sera 2','Ref S1','Ref S2','Min','Max');
title('Evoluția umidității');
grid on;

figure;
plot(0:T_sim-1, u_traj(1,:), '-', 'LineWidth', 2.2); hold on;
plot(0:T_sim-1, u_traj(2,:), '-', 'LineWidth', 2.2);
yline(0, '--k');
xlabel('Timp [pas]');
ylabel('Comandă');
legend('u_1', 'u_2', 'Zero');
title('Comenzi aplicate');
grid on;
