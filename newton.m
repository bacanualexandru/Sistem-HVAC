% Parametri
epsilon = 1e-3;
sigma = 0.5;

for k = 1:T_sim
    % Construiește problema QP
    [H, q, C, d] = denseMpcBariera(A, B, Q, R, x0, N, u_ub, u_lb, x_ref, u_ref);
    x = zeros(size(H,1), 1); % variabila de decizie
    tau = 1;
    m = size(C,1); % nr constrângeri

    while m * tau >= epsilon
        % Calculează gradientul și hessiana cu barieră
        ci = C * x - d;
        b_grad = zeros(size(x));
        b_hess = zeros(size(H));
        for i = 1:m
            b_grad = b_grad + C(i,:)' / ci(i);
            b_hess = b_hess + (C(i,:)' * C(i,:)) / (ci(i)^2);
        end
        grad = H * x + q - tau * b_grad;
        hess = H + tau * b_hess;
        % Pas Newton
        d_x = - hess \ grad;
        % Line search simplu
        alpha = 1e-3;
        while true
            x_new = x + alpha * d_x;
            if all(C * x_new - d < 0), break; end
            alpha = alpha * 0.5; % micșorez pasul dacă sunt în afara domeniului
        end
        x = x + alpha * d_x;
        tau = tau * sigma;
    end

    % Aplică comanda optimă și actualizează starea
    u_applied = x(1:m); % extrag primul control
    x0 = A * x0 + B * u_applied + disturbance_traj(:,k);

    % Salvează traiectoriile
    x_traj(:,k+1) = x0;
    u_traj(:,k) = u_applied + disturbance_traj(:,k);
end
