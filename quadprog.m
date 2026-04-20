clear; clc;

% ---------------------
% Parametri model sistem sere
% ---------------------
A = [0.9 0.1;
     0.1 0.9];
B = [0.2 0.1;
     0.1 0.2];

x_ref = [0.6; 0.5];    % ținte umiditate
u_ref = [0; 0];        % referință comandă (zero)

N = 30;                % orizont de predicție
lambda = 0.1;          % penalizare pe comenzi
mu = 0.3;              % (nefolosit aici, doar pentru CVX/penalizare variații)

T_sim = 17;            % durată simulare
x0 = [0.45; 0.4];      % stare inițială

n = size(A, 1);
m = size(B, 2);

Q = eye(n);
R = lambda * eye(m);

x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);

u_ub = ones(m, 1);     % limită superioară comenzi
u_lb = -ones(m, 1);    % limită inferioară comenzi

% Perturbații
rng(1);
disturbance_traj = [
    0.001  0.001  0.002  -0.03  0.002  0.002  0.002  0.001  0.003  0.002  ...
    0.001  0.001  0.002  0.002  0.01  0.001  0.002;
    0.000  0.001  0.002  0.002  0.001  0.001  0.001  0.000  0.001  0.002  ...
    0.002  0.001  0.001  0.001  0.002  0.002  0.001
];

% ---------------------
% Bucla MPC + QUADPROG
% ---------------------
for k = 1:T_sim
    [H, q, C, d] = denseMpcBariera(A, B, Q, R, x0, N, u_ub, u_lb, x_ref, u_ref);
    % Solve QP cu quadprog
    options = optimoptions('quadprog', 'Display', 'off');
    H_sym = (H + H') / 2;
    U_opt = quadprog(2*H, q, C, d, [], [], [], [], [], options);
    % Aplică primul control
    u_applied = U_opt(1:m);
    x0 = A * x0 + B * u_applied + disturbance_traj(:,k);
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied + disturbance_traj(:,k);
end

% ---------------------
% GRAFICE rezultate
% ---------------------
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
title('Evoluția umidității (quadprog)');
grid on;

figure;
plot(0:T_sim-1, u_traj(1,:), '-', 'LineWidth', 2.2); hold on;
plot(0:T_sim-1, u_traj(2,:), '-', 'LineWidth', 2.2);
yline(0, '--k');
xlabel('Timp [pas]');
ylabel('Comandă');
legend('u_1', 'u_2', 'Zero');
title('Comenzi aplicate (quadprog)');
grid on;
