clear; clc;
A = [0.9 0.1; 0.1 0.9];
B = [0.2 0.1; 0.1 0.2];
x_ref = [0.6; 0.5];
u_ref = [0; 0];
N = 30; lambda = 0.1; mu = 0.3; T_sim = 17;
x0 = [0.45; 0.4];
n = size(A,1); m = size(B,2);
Q = eye(n); R = lambda * eye(m);
u_ub = ones(m,1); u_lb = -ones(m,1);
rng(1);
disturbance_traj = [...
    0.001  0.001  0.002  -0.03  0.002  0.002  0.002  0.001  0.003  0.002  ...
    0.001  0.001  0.002  0.002  0.01  0.001  0.002;
    0.000  0.001  0.002  0.002  0.001  0.001  0.001  0.000  0.001  0.002  ...
    0.002  0.001  0.001  0.001  0.002  0.002  0.001
];

[x_cvx, u_cvx] = run_cvx(A,B,Q,R,x0,x_ref,lambda,mu,N,T_sim,disturbance_traj);
[x_grad, u_grad,all_costs_grad] = run_barrier_gradient(A,B,Q,R,x0,x_ref,u_ref,N,T_sim,u_ub,u_lb,disturbance_traj);
[x_newton, u_newton,all_costs_newton] = run_barrier_newton(A,B,Q,R,x0,x_ref,u_ref,N,T_sim,u_ub,u_lb,disturbance_traj);
t = 0:T_sim;
figure;
plot(t, x_cvx(1,:), 'b-', 'LineWidth', 2); hold on;
plot(t, x_grad(1,:), 'r--', 'LineWidth', 2);
plot(t, x_newton(1,:), 'g-.', 'LineWidth', 2);
yline(x_ref(1), 'k:', 'LineWidth', 1.5);
xlabel('Timp [pas]'); ylabel('Umiditate - Sera 1');
title('Comparare metode - Sera 1');
legend('CVX','Gradient','Newton','Ref'); grid on;
figure;
plot(t, x_cvx(2,:), 'b-', 'LineWidth', 2); hold on;
plot(t, x_grad(2,:), 'r--', 'LineWidth', 2);
plot(t, x_newton(2,:), 'g-.', 'LineWidth', 2);
yline(x_ref(2), 'k:', 'LineWidth', 1.5);
xlabel('Timp [pas]'); ylabel('Umiditate - Sera 2');
title('Comparare metode - Sera 2');
legend('CVX','Gradient','Newton','Ref'); grid on;
figure;
plot(0:T_sim-1, u_cvx(1,:), 'b-', 'LineWidth', 2); hold on;
plot(0:T_sim-1, u_grad(1,:), 'r--', 'LineWidth', 2);
plot(0:T_sim-1, u_newton(1,:), 'g-.', 'LineWidth', 2);
yline(0, 'k:');
xlabel('Timp'); ylabel('u_1'); title('Comparare comenzi u_1');
legend('CVX','Gradient','Newton','Zero'); grid on;
figure;
plot(0:T_sim-1, u_cvx(2,:), 'b-', 'LineWidth', 2); hold on;
plot(0:T_sim-1, u_grad(2,:), 'r--', 'LineWidth', 2);
plot(0:T_sim-1, u_newton(2,:), 'g-.', 'LineWidth', 2);
yline(0, 'k:');
xlabel('Timp'); ylabel('u_2'); title('Comparare comenzi u_2');
legend('CVX','Gradient','Newton','Zero'); grid on;
figure;
for t = 1:T_sim
    plot(1:length(all_costs_grad{t}), all_costs_grad{t}, 'r-', 'LineWidth', 2.5); hold on;
    plot(1:length(all_costs_newton{t}), all_costs_newton{t}, 'g.-', 'LineWidth', 2.5, 'MarkerSize', 10);
end
xlabel('Iterație');
ylabel('Normă gradient');
title('Convergența internă per pas');
legend('Gradient', 'Newton');
grid on;
% Calcul MSE pentru fiecare metodă
mse_cvx = mean(vecnorm(x_cvx - x_ref).^2);
mse_grad = mean(vecnorm(x_grad - x_ref).^2);
mse_newton = mean(vecnorm(x_newton - x_ref).^2);

% Afișare în consolă
fprintf('Eroare medie pătratică (MSE):\n');
fprintf('CVX     : %.6f\n', mse_cvx);
fprintf('Gradient: %.6f\n', mse_grad);
fprintf('Newton  : %.6f\n', mse_newton);

methods = {'CVX', 'Gradient', 'Newton'};
mse_vals = [mse_cvx, mse_grad, mse_newton];

figure;
bar(mse_vals);
set(gca, 'XTickLabel', methods);
ylabel('Eroare medie pătratică (MSE)');
title('Comparare MSE metode');
grid on;
