clear;
clc;

% sistemul (model liniar 2 sere)
A = [0.9 0.1;
     0.1 0.9];
B = [0.2 0.1;
     0.1 0.2];

x_ref = [0.6; 0.5]; % tinta umiditate
N = 30;            % orizont predictie MPC
lambda = 0.1;      % penalizare pe u
mu = 0.3;          % penalizare pe delta u
T_sim = 17;        % durata simulare
x0 = [0.45; 0.4];  % stare initiala

n = size(A,1);
m = size(B,2);

% initializare traiectorii
x_traj = zeros(n, T_sim+1); x_traj(:,1) = x0;
u_traj = zeros(m, T_sim);

% perturbatii (zgomot realist)
rng(1);
disturbance_traj = [
    0.001  0.001  0.002  -0.03  ... 0;
    0.000  0.001  0.002   0.002 ... 0
];

% MPC cu fmincon la fiecare pas
for k = 1:T_sim
    U0 = zeros(N * m, 1);         % guess initial
    lb = -1 * ones(N * m, 1);     % limite comenzi
    ub =  1 * ones(N * m, 1);
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    % optimizeaza folosind fmincon
    U_opt = fmincon(@(U) cost_fun(U, x0, A, B, x_ref, lambda, mu, N), ...
                    U0, [], [], [], [], lb, ub, ...
                    @(U) constraints_fun(U, x0, A, B, N), ...
                    options);
    % aplic prima comanda
    u_applied = U_opt(1:m);
    x0 = A * x0 + B * u_applied + disturbance_traj(:,k);
    % salveaza stare si comanda
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied + disturbance_traj(:,k);
end

% === GRAFIC STARI ===
t = 0:T_sim;
figure;
plot(t, x_traj(1,:), '-', 'LineWidth', 2.2); hold on;
plot(t, x_traj(2,:), '-', 'LineWidth', 2.2);
yline(0.6, '--', 'LineWidth', 1.5); % ref S1
yline(0.5, '--', 'LineWidth', 1.5); % ref S2
yline(0.4, '-', 'LineWidth', 1);   % limita min
yline(0.7, '-', 'LineWidth', 1);   % limita max

xlabel('Timp [pas]');
ylabel('Umiditate');
legend('Sera 1','Sera 2','Ref S1','Ref S2','Min','Max');
title('Evolutia umiditatii');
grid on;

% === GRAFIC COMENZI ===
figure;
plot(0:T_sim-1, u_traj(1,:), '-', 'LineWidth', 2.2); hold on;
plot(0:T_sim-1, u_traj(2,:), '-', 'LineWidth', 2.2);
yline(0, '--k');
xlabel('Timp [pas]');
ylabel('Comanda');
legend('u_1', 'u_2', 'Zero');
title('Comenzi aplicate');
grid on;

% === FUNCTIE COST ===
function J = cost_fun(U, x0, A, B, x_ref, lambda, mu, N)
    n = length(x0);
    m = length(U) / N;
    X = zeros(n, N+1);
    X(:,1) = x0;
    U_mat = reshape(U, m, N);
    for k = 1:N
        X(:,k+1) = A * X(:,k) + B * U_mat(:,k);
    end
    % cost total = distanta + efort + variatie
    J = 0;
    for k = 1:N
        J = J + norm(X(:,k) - x_ref)^2 + lambda * norm(U_mat(:,k))^2;
    end
    for k = 1:N-1
        J = J + mu * norm(U_mat(:,k+1) - U_mat(:,k))^2;
    end
end
% === CONSTRANGERI STARE ===
function [c, ceq] = constraints_fun(U, x0, A, B, N)
    n = length(x0);
    m = length(U) / N;
    X = zeros(n, N+1);
    X(:,1) = x0;
    U_mat = reshape(U, m, N);
    for k = 1:N
        X(:,k+1) = A * X(:,k) + B * U_mat(:,k);
    end
    % constrangeri de tip: x in [0.4, 0.7]
    c = [];
    for k = 1:N
        c = [c; X(:,k) - 0.7; 0.4 - X(:,k)];
    end
    ceq = []; % fara egalitati
end
