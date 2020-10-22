% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSID: Preferential Subspace Identification Algorithm
% Identifies a linear stochastic model for a signal y, while prioritizing
% the latent states that are predictive of another signal z. The model is
% as follows:
% [x1(k+1); x2(k+1)] = [A11 0; A21 A22] * [x1(k); x2(k)] + w(k)
%               y(k) =      [Cy1   Cy2] * [x1(k); x2(k)] + v(k)
%               z(k) =      [Cz1     0] * [x1(k); x2(k)] + e(k)
% x(k) = [x1(k); x2(k)] => Latent state time series
% x1(k) => Latent states related to z ( the pair (A11, Cz1) is observable )
% x2(k) => Latent states unrelated to z 
% Given training time series from y(k) and z(k), the dimension of x(k) 
% (i.e. nx), and the dimension of x1(k) (i.e. n1), the algorithm finds 
% all model parameters and noise statistics:
%       - A  : [A11 0; A21 A22]
%       - Cy : [Cy1   Cy2]
%       - Cz : [Cz1     0]
%       - Q  : Cov( w(k), w(k) )
%       - R  : Cov( v(k), v(k) )
%       - S  : Cov( w(k), v(k) )
% as well as the following model characteristics/parameters: 
%       - G  : Cov( x(k+1), y(k) )
%       - YCov: Cov( y(k), y(k) )
%       - K: steady state stationary Kalman filter for estimating x from y
%       - innovCov: covariance of innovation for the Kalman filter
%       - P: covariance of Kalman predicted state error
%       - xPCov: covariance of Kalman predicted state itself
%       - xCov: covariance of the latent state
%   
%   Inputs:
%     - (1) y: Inputs signal 1 (e.g. neural signal). 
%               Must be ny x T:
%               [y(1), y(2), y(3), ..., y(T)]
%     - (2) z: Inputs signal 2, to be studied using y (e.g. behavior). 
%               Must be nz x T:
%               [z(1), z(2), z(3), ..., z(T)]
%     - (3) nx: the total number of latent states in the stochastic model
%     - (4) n1: number of latent states to extract in the first stage.
%     - (5) i: the number of block-rows (i.e. future and past horizon) 
%     - (6) ws (optional): the ws output from a previous call using the exact 
%               same data. If calling PSID_QRS repeatedly with the same data 
%               and horizon, several computationally costly steps can be 
%               reused from before. Otherwise will be discarded.
%   Outputs:
%     - (1) idSys: structure with the system parameters for the identified
%           system. Will have the following fields (defined above):
%               'A', 'Cy', 'Cz', 'Q', 'R', 'S'
%               'G', 'YCov', 'K', 'innovCov', 'P', 'xPCov', 'xCov' 
%     - (2) ws (optional): structure to provide to later calls of PSID /
%               on the same data (see input (6) for more details)
%   Usage example:
%       idSys = PSID(y, z, nx, n1, i);
%       [idSys, WS] = PSID(y, z, nx, n1, i, WS);
%       idSysSID = PSID(y, z, nx, 0, i); % Set n1=0 for SID

function [idSys, WS] = PSID(y, z, nx, n1, i, WS)

if nargin < 6, WS = struct; end
if ~isstruct(WS), WS = struct; end

[ny, ySamples, N, y1] = getHSize(y, i);
[nz, zSamples, ~, z1] = getHSize(z, i);

if isfield(WS, 'N') && isequal(WS.N, N) ...
    && isfield(WS, 'k') && isequal(WS.k, i) ...
    && isfield(WS, 'ny') && isequal(WS.ny, ny) ...
    && isfield(WS, 'ySamples') && isequal(WS.ySamples, ySamples) ...
    && isfield(WS, 'y1') && isequal(WS.y1, y1) ...
    && isfield(WS, 'nz') && isequal(WS.nz, nz) ...
    && isfield(WS, 'zSamples') && isequal(WS.zSamples, zSamples) ...
    && (isempty(z) || (isfield(WS, 'z1') && isequal(WS.z1, z1)))
    % Have WS from previous call with the same data
else
    WS = struct( ...
        'N', N, ...
        'k', i, ...
        'ny', ny, ...
        'nz', nz, ...
        'ySamples', ySamples, ...
        'zSamples', zSamples, ...
        'y1', y1 ...
    );
    if ~isempty(z), WS.z1 = z1; end
end

if ~isfield(WS, 'Yp') || isempty(WS.Yp)
    WS.Yp = blkhankskip(y, i, N);
    WS.Yii = blkhankskip(y, 1, N, i);
    if nz > 0
        WS.Zii = blkhankskip(z, 1, N, i);
    end
end

if n1 > nx, n1 = nx; end % n1 can at most be nx

% Stage 1
if n1 > 0 && nz > 0
    if ~isfield(WS, 'ZHat_U') || isempty(WS.ZHat_U)
        % Zp = blkhankskip(z, k, N);
        Zf = blkhankskip(z, i, N, i);
        WS.ZHat = (Zf / WS.Yp) * WS.Yp; % Zf * WS.Yp.' * pinv(WS.Yp * WS.Yp.') * WS.Yp;  % Eq. (10)
        
        Yp_Plus = [WS.Yp; WS.Yii];
        Zf_Minus = Zf((nz+1):end, :);
        WS.ZHatMinus = (Zf_Minus / Yp_Plus) * Yp_Plus; % Zf_Minus * Yp_Plus.' * pinv(Yp_Plus * Yp_Plus.') * Yp_Plus;  % Eq. (11)

        % Take SVD of ZHat
        [WS.ZHat_U,WS.ZHat_S,~] = svd(WS.ZHat, 'econ');  % Eq. (12)
        
        clear Zf Zf_Minus Yp_Plus % Free up mem
    end

    Sz = WS.ZHat_S(1:n1, 1:n1);                 % Eq. (12)
    Uz = WS.ZHat_U( :  , 1:n1);                 % Eq. (12)
    
    Oz = Uz * Sz^(1/2);                         % Eq. (13)
    Oz_Minus = Oz(1:(end-nz), :);               % Eq. (15)
    
    Xk = pinv(Oz) * WS.ZHat;                    % Eq. (14)
    Xk_Plus1 = pinv(Oz_Minus) * WS.ZHatMinus;   % Eq. (16)
else
    n1 = 0;
    Xk = [];
    Xk_Plus1 = [];
end

% Stage 2
n2 = nx - n1;
if n2 > 0
    if ~isfield(WS, 'YHat_U') || isempty(WS.YHat_U) || ~isfield(WS, 'n1') || ~isequal(WS.n1, n1)
        WS.n1 = n1; % Will store RYfYp after subtracting the prediction of the first n1 states 
        
        Yf = blkhankskip(y, i, N, i);
        Yf_Minus = Yf((ny+1):end, :);
        
        if n1 > 0 % Have already extracted some states, so remove the already predicted part of Yf
            % Remove the already predicted part of future y
            Oy1 = Yf / Xk;         % Eq. (18) - Find the y observability matrix for Xk
            Yf = Yf - Oy1 * Xk;    % Eq. (19)
            
            Oy1_Minus = Oy1(1:(end-ny), :);                 % Eq. (20)
            Yf_Minus = Yf_Minus - Oy1_Minus * Xk_Plus1;     % Eq. (21)
        end
        
        WS.YHat = (Yf / WS.Yp) * WS.Yp; % Yf * WS.Yp.' * pinv(WS.Yp * WS.Yp.') * WS.Yp;  % Eq. (22)
        
        Yp_Plus = [WS.Yp; WS.Yii];
        WS.YHatMinus = (Yf_Minus / Yp_Plus) * Yp_Plus; % Yf_Minus * Yp_Plus.' * pinv(Yp_Plus * Yp_Plus.') * Yp_Plus;  % Eq. (23)
        
        % Take SVD of YHat
        [WS.YHat_U,WS.YHat_S,~] = svd(WS.YHat, 'econ');     % Eq. (24)
        
        clear Yf Yf_Minus Yp_Plus % Free up mem
    end
    
    S2 = WS.YHat_S(1:n2, 1:n2);                  % Eq. (24)
    U2 = WS.YHat_U( :  , 1:n2);                  % Eq. (24)
    
    Oy = U2 * S2^(1/2);                          % Eq. (25)
    Oy_Minus = Oy(1:(end-ny), :);                % Eq. (27)
    
    Xk2 = pinv(Oy) * WS.YHat;                    % Eq. (26)
    Xk2_Plus1 = pinv(Oy_Minus) * WS.YHatMinus;   % Eq. (28)
    
    Xk = cat(1, Xk, Xk2);                        % Eq. (29)
    Xk_Plus1 = cat(1, Xk_Plus1, Xk2_Plus1);      % Eq. (29)
    clear Xk2 Xk2_Plus1 % Free up mem
end

if n1 > 0
    % A associated with the z-related states
    A = Xk_Plus1(1:n1, :) / Xk(1:n1, :);        % Eq. (17)
else
    A = [];
end

if n2 > 0
    % A associated with the other states
    A23 = Xk_Plus1((n1+1):end, :) / Xk(:, :);   % Eq. (30)
    A = [A, zeros(n1, n2); A23];                % Eq. (31)
end

w = Xk_Plus1 -  A * Xk;                         % Eq. (34)

if nz > 0
    Cz = WS.Zii / Xk;                           % Eq. (33)
else
    Cz = [];
end

Cy = WS.Yii / Xk;                               % Eq. (32)
v  = WS.Yii - Cy * Xk;                          % Eq. (34)

clear Xk Xk_Plus1 % Free up mem

% Compute noise covariances 
NA = size(w, 2);
Q = (w * w.')/NA;                               % Eq. (35)
S = (w * v.')/NA;                               % Eq. (35)
R = (v * v.')/NA;                               % Eq. (35)

Q = (Q + Q.')/2; % Make precisely symmetric
R = (R + R.')/2; % Make precisely symmetric

% Compute secondary system matrices, e.g. Kalman gain, etc
[K, P, innovCov] = computeKalmanParamsFromQRS( A, Cy, Q, R, S );
[G, YCov, xCov] = computeGCovParamsFromQRS( A, Cy, Q, R, S );
xPCov = xCov - P;       % Covariance of Kalman predicted states

idSys = struct( ...
    'A', A, ...
    'Cy', Cy, ...
    'Cz', Cz, ...
    'Q', Q, ...
    'R', R, ...
    'S', S, ...
    'G', G, ...
    'YCov', YCov, ...
    'K', K, ...
    'innovCov', innovCov, ...
    'P', P, ...
    'xPCov', xPCov, ...
    'xCov', xCov ...
);

end

function [ny, ySamples, N, y1] = getHSize(y, i)

ny = nan; y1 = nan; 
if ~iscell(y)
    [ny, ySamples] = size(y); N = ySamples - 2*i + 1;
    if ~isempty(y)
        y1 = y(1);
    end
else
    ySamples = [];
    N = [];
    for yInd = 1:numel(y)
        [nyThis, ySamplesThis, NThis, y1This] = getHSize(y{yInd}, i);
        if yInd == 1
            ny = nyThis;
            y1 = y1This;
        else
            if nyThis ~= ny
                error('Size of dimension 1 must be the same in all elements of cell array.');
            end
        end
        ySamples = cat(1, ySamples, ySamplesThis);
        N = cat(1, N, NThis);
    end
end
    
end