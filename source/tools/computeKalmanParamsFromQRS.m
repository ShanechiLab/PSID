% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computeKalmanParamsFromQRS Computes the Kalman filtering parameters
%   Inputs:
%     - (1) A: State transfer matrix
%     - (2) C: Observation matrix
%     - (3) Q: state noise covariance
%     - (4) R: observation noise covariance
%     - (5) S: state-observation noise cross covariance
%   Outputs:
%     - (1) K: steady state stationary Kalman filter for estimating x from y
%     - (2) innovCov: covariance of innovation for the Kalman filter
%     - (3) P: covariance of Kalman predicted state error
%   Usage example:
%       [K, P, innovCov] = computeKalmanParamsFromQRS( A, C, Q, R, S )

function [K, P, innovCov] = computeKalmanParamsFromQRS( A, C, Q, R, S )

try
    [P,~,GG] = dare(A',C',Q,R,S); % Solves eq. 47
    K = GG'; % Total K = A * Kf + Kv

    innovCov = C*P*C.' + R;

    % Kf = P*C.'/innovCov;
    % Kv = S/innovCov;
catch ME
    fprintf('WARNING: %s\n', ME.message);
    P = nan(size(A, 1));
    K = nan(size(A, 1), size(C, 1));
    innovCov = nan(size(C, 1));
end

end
