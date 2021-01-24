% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generateLSSMRealization Generates a realization of an LSSM with the given
% parameters, with gaussian state and observation noises with the given 
% covariance matrices
%   Inputs:
%     - (1) csys: LSSM system struct. Contanis fields:
%                   'a': The A matrix
%                   'c': The C matrix
%                 And either:
%                   'q': The Q matrix -> state noise covariance matrix
%                   'r': The R matrix -> observation noise covariance 
%                   's': The S matrix -> matrix state and observation noise
%                                        cross-covariance matrix
%                 Or to generate realization from innovations formulation:
%                   'K': Kalman gain
%                   'innovCov': innovation covariance
%     - (2) t:     time indexes of samples to generate, for now only the
%                  length of this vector has an effect. 
%     - (3) X0:    latest system state at t = t0 - 1;
%     - (4) W0:    latest state noise at t = t0 - 1; % Must have had proper
%                  covariance with output noise previously generated at t = t0-1
%   Outputs:
%     - (1) y: Observations in the generated realization
%     - (2) x (optional): Value of the hidden state in the generated realization
%     - (3) t (optional): time sample values 
%     - (4) xN (optional): The last state for time t (useful for continuing 
%                                     the same realization in a later call)
%     - (5) wN (optional): The last state noise for time t (useful for 
%                          continuing the same realization in a later call)
%     - (6) w (optional): state noise
%     - (7) v (optional): observaion noise
%   Usage example:
%       [y,x,t] = generateLSSMRealization(trueSys, 1:10000);

function [y, x, t, xN, wN, w, v] = generateLSSMRealization(csys, t, X0, W0)

if nargin < 3, X0 = []; end
if nargin < 4, W0 = []; end

if ~isfield(csys, 'a'), csys.a = []; end
if ~isfield(csys, 'c'), csys.c = []; end
nx = size(csys.a, 1);

if ~isfield(csys,'s'), csys.s = []; end

Q = csys.q;
R = csys.r;
S = csys.s;

if isempty(Q), Q = zeros(size(csys.a, 1)); end
if isempty(R), R = zeros(size(csys.c, 1)); end
if isempty(S), S = zeros(size(Q, 1), size(R, 1)); end

% Combine noise covariances
L = [Q, S; S', R];
[eigVecs,eigVals] = eig(L);
% Prepare shaping matrices for model and observation noise
noiseShaping = eigVecs*sqrt(eigVals);

allNoise = randn(length(t), size(eigVals, 1))*noiseShaping';

w = allNoise(:, 1:nx);
v = allNoise(:, (nx+1):end);

ny = size(v, 2);

x = nan(length(t), nx);
y = nan(length(t), ny);

if isempty(X0), X0 = zeros(nx, 1); end
if isempty(W0), W0 = zeros(nx, 1); end

if nx == 0
    y = v;
else
    for ti = 1:length(t)
        if ti == 1
            Xt_1 = X0(:);
            Wt_1 = W0(:);
        else
            Xt_1 = x(ti-1, :)';
            Wt_1 = w(ti-1, :)';
        end
        x(ti, :) = csys.a*Xt_1 + Wt_1;
        y(ti, :) = csys.c*x(ti, :)' + v(ti, :)';
    end
end

xN = x(end, :)'; 
wN = w(end, :)'; 

end