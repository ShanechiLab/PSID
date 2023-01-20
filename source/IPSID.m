% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IPSID: Input Preferential Subspace Identification Algorithm
% IPSID identifies a linear stochastic model for a signal y, while prioritizing
% the latent states that are predictive of another signal z, while a known
% external input u is applied to the system.
% The complete model is as follows:
% [x1(k+1); x2(k+1)] = [A11 0; A21 A22] * [x1(k); x2(k)] + [B1; B2] * u(k) + w(k)
%               y(k) = [Cy1   Cy2] * [x1(k); x2(k)] + [Dy] * u(k) + v(k)
%               z(k) = [Cz1     0] * [x1(k); x2(k)] + [Dz] * u(k) + e(k)
% x(k) = [x1(k); x2(k)] => Latent state time series
% x1(k) => Latent states related to z ( the pair (A11, Cz1) is observable )
% x2(k) => Latent states unrelated to z 
% u(k) => External input that was applied to the system
% Given training time series from y(k), z(k) and u(k), the dimension of x(k) 
% (i.e. nx), and the dimension of x1(k) (i.e. n1), the algorithm finds 
% all model parameters and noise statistics:
%       - A  : [A11 0 0; A21 A22 0]
%       - Cy : [Cy1   Cy2]
%       - Cz : [Cz1     0]
%       - B  : [B1     B2]
%       - Dy : [Dy]
%       - Dz : [Dz]
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
% and the following z residual statistics (which would not be needed for
% decoding z using y:
%       - Sxz : Cov( w(k), e(k) )
%       - Syz : Cov( v(k), e(k) )
%       - Rz  : Cov( e(k), e(k) )
%   
%   Inputs:
%     - (1)  y: Inputs signal 1 (e.g. neural signal). 
%               Must be ny x T:
%               [y(1), y(2), y(3), ..., y(T)]
%     - (2)  z: Inputs signal 2, to be studied using y (e.g. behavior). 
%               Must be nz x T:
%               [z(1), z(2), z(3), ..., z(T)]
%     - (3)  u: Inputs signal 3, signal driving y and z (e.g. system input). 
%               Must be nu x T:
%               [u(1), u(2), u(3), ..., u(T)]
%     - (4)  nx: the total number of latent states in the stochastic model
%     - (5)  n1: number of behavior-related latent states. 
%               For n1 = 0, will be equivalent to SID, due to 
%               VODM Book, Stochastic algorithm 3
%     - (6)  k: the number of block-rows (i.e. future and past horizon) 
%     - (7)  ws: the ws output from a previous call using the exact same 
%               data. If calling IPSID repeatedly with the same data and
%               horizon, several computationally costly steps can be 
%               reused from before. Otherwise will be discarded.
%     - (8)  fit_Cz_via_KF (default: true): if true (preferred option), 
%               refits Cz more accurately using a KF after all other 
%               paramters are learned
%     - (9) remove_mean_Y: if true will remove the mean of y. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (10) remove_mean_Z: if true will remove the mean of z. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (11) remove_mean_U: if true will remove the mean of u. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (12) zscore_Y: if true will z-score y. It is ok to set this to false,
%                     but setting to true may help with stopping some dimensions of 
%                     data from dominating others. Defaults to false.
%     - (13) zscore_Z: if true will z-score Z. It is ok to set this to false,
%                     but setting to True may help with stopping some dimensions of 
%                     data from dominating others. Defaults to true.
%     - (14) zscore_U: if true will z-score U. It is ok to set this to false,
%                     but setting to True may help with stopping some dimensions of 
%                     data from dominating others. Defaults to true.
%   Outputs:
%     - (1) idSys: structure with the system parameters for the identified
%           system. Will have the following fields (defined above):
%               'A', 'Cy', 'Cz', 'Q', 'R', 'S'
%               'B', 'Dy', 'Dz'
%               'G', 'YCov', 'K', 'innovCov', 'P', 'xPCov', 'xCov' 
% 
%   Usage example:
%       idSys = IPSID(y, z, u, nx, n1, k);
%       [idSys, WS] = IPSID(y, z, u, nx, n1, k, 'WS',WS); 
%       idSysISID = IPSID(y, z, u, nx, 0, k); % Set n1=0 for ISID
%       idSysPSID = IPSID(y, z, [], nx, n1, k); % Set u=[] for PSID
%       idSysSID = IPSID(y, z, [], nx, 0, k); % Set n1=0 and u=[] for SID

function [idSys, WS] = IPSID(y, z, u, nx, n1, k, varargin)

p = inputParser;

addParameter(p, 'WS', struct, @isstruct);
addParameter(p, 'fit_Cz_via_KF', true, @islogical);

addParameter(p, 'remove_mean_Y', true, @islogical);
addParameter(p, 'remove_mean_Z', true, @islogical);
addParameter(p, 'remove_mean_U', true, @islogical);
addParameter(p, 'zscore_Y', false, @islogical);
addParameter(p, 'zscore_Z', false, @islogical);
addParameter(p, 'zscore_U', false, @islogical);

parse(p,varargin{:});

WS = p.Results.WS;
fit_Cz_via_KF = p.Results.fit_Cz_via_KF;

YPrepModel = PrepModel();
YPrepModel.fit(y, 'remove_mean', p.Results.remove_mean_Y, 'zscore', p.Results.zscore_Y, 'time_first', time_first);
y = YPrepModel.apply(y, time_first);

ZPrepModel = PrepModel();
if ~isempty(z)
    ZPrepModel.fit(z, 'remove_mean', p.Results.remove_mean_Z, 'zscore', p.Results.zscore_Z, 'time_first', time_first);
    z = ZPrepModel.apply(z, time_first);
end

UPrepModel = PrepModel();
if ~isempty(u)
    UPrepModel.fit(u, 'remove_mean', p.Results.remove_mean_U, 'zscore', p.Results.zscore_U, 'time_first', time_first);
    u = UPrepModel.apply(u, time_first);
end

[ny, ySamples, N, y1] = getHSize(y, k, time_first);
[nz, zSamples, ~, z1] = getHSize(z, k, time_first);
[nu, uSamples, ~, u1] = getHSize(u, k, time_first);

requiredSamples = ((2*nu+2*ny)*k)+2*k-1;
if ySamples < requiredSamples && nu > 0
    error('Too few training samples! Identification with ny=%d, nu=%d, k=%d requires at least %d samples, but recieved %d!', ny, nu, k, requiredSamples, ySamples)
end

if isfield(WS, 'N') && isequal(WS.N, N) && isfield(WS, 'k') && isequal(WS.k, k) ...
    && isfield(WS, 'ySize') && isequal(WS.ySize, size(y)) ...
    && isfield(WS, 'zSize') && isequal(WS.zSize, size(z)) ...
    && isfield(WS, 'y1') && isequal(WS.y1, y(1)) ...
    && (isempty(z) || (isfield(WS, 'z1') && isequal(WS.z1, z(1))))
    % Have WS from previous call with the same data
else
    WS = struct( ...
        'N', N, ...
        'k', k, ...
        'ySize', size(y), ...
        'y1', y(1), ...
        'zSize', size(z) ...
    );
    if ~isempty(z), WS.z1 = z(1); end
end

if ~isfield(WS, 'Yp') || isempty(WS.Yp)
    WS.Yp = blkhankskip(y, k, N, 0, time_first);
    WS.Yf = blkhankskip(y, k, N, k, time_first);
    WS.Yii = blkhankskip(y, 1, N, k, time_first);
    if nu > 0
        WS.Up = blkhankskip(u, k, N, 0, time_first);
        WS.Uf = blkhankskip(u, k, N, k, time_first);
        WS.Uii = blkhankskip(u, 1, N, k, time_first);
    else
        WS.Up = [];
        WS.Uf = [];
        WS.Uii = [];
    end
    if nz > 0
        WS.Zii = blkhankskip(z, 1, N, k, time_first);
    end
end

if n1 > nx % n1 can at most be nx
    n1 = nx;
end 

if n1 > 0 && nz > 0
    if ~isfield(WS, 'ZHatObUfRes_U') || isempty(WS.ZHatObUfRes_U) 
        Zf = blkhankskip(z, k, N, k, time_first);
        Zf_Minus = Zf((nz+1):end, :);
        Uf_Minus = WS.Uf((nu+1):end, :);
           
        % IPSID Stage 1:
        % -----------------------
        % Oblique projection of Zf along Uf, onto [Up;Yp]
        ZHatOb = projOblique(Zf, [WS.Up; WS.Yp], WS.Uf); 
        WS.ZHatObUfRes = removeProjOrth(ZHatOb, WS.Uf);
        
        % Orthogonal projection of Zf onto [Up;Yp;Uf]
        WS.ZHat = projOrth(Zf, [WS.Up; WS.Yp; WS.Uf]);

        % Orthogonal projection of Zf_Minus onto [Up_plus;Yp_plus;Uf_Minus]
        WS.ZHatMinus = projOrth(Zf_Minus, [WS.Up; WS.Uii; WS.Yp; WS.Yii; Uf_Minus]);

        % Take SVD of ZHatObUfRes
        [WS.ZHatObUfRes_U,WS.ZHatObUfRes_S,~] = svd(WS.ZHatObUfRes, 'econ');
        
        clear Zf Zf_Minus Uf_Minus ZHatOb % Free up memory
    end

    Sz = WS.ZHatObUfRes_S(1:n1, 1:n1); 
    Uz = WS.ZHatObUfRes_U( :  , 1:n1);
    
    Oz = Uz * Sz^(1/2);                         
    Oz_Minus = Oz(1:(end-nz), :);               
    
    Xk = pinv(Oz) * WS.ZHat;                   
    Xk_Plus1 = pinv(Oz_Minus) * WS.ZHatMinus;   
   
else
    n1 = 0;
    Xk = [];
    Xk_Plus1 = [];
end


n2 = nx - n1;

% IPSID Stage 2:
% -----------------------
if n2 > 0
    if ~isfield(WS, 'YHatObUfRes_U') || isempty(WS.YHatObUfRes_U) || ~isfield(WS, 'n1') || ~isequal(WS.n1, n1)
        WS.n1 = n1; 
   
        Yf = WS.Yf;
        Yf_Minus = WS.Yf((ny+1):end, :);
        Uf_Minus = WS.Uf((nu+1):end, :);
        
        if n1 > 0 % Have already extracted some states (x1), so remove the already predicted part of Yf
                  % Remove the already predicted part of future y (while keeping the part predicted from Up and Uf)
                  % Oblique projection of Yf along Uf, onto Wp
            [YHatOb1, Oy1] = projOblique(Yf, Xk, [WS.Up; WS.Uf]); 
            Yf = Yf - YHatOb1; 
            clear YHatOb1
            
            Oy1_Minus = Oy1(1:(end-ny), :);     
            Yf_Minus = Yf_Minus - Oy1_Minus * Xk_Plus1;  
        end
        
        % Oblique projection of Yf along Uf, onto [Up;Yp]: 
        YHatOb = projOblique(Yf, [WS.Up; WS.Yp], WS.Uf); 
        WS.YHatObUfRes = removeProjOrth(YHatOb, WS.Uf);
        
        % Orthogonal projection of Yf onto [Up;Yp;Uf]
        WS.YHat = projOrth(Yf, [WS.Up; WS.Yp; WS.Uf]);

        % Orthogonal projection of Yf_Minus onto[Up_plus;Yp_plus;Uf_Minus]
        WS.YHatMinus = projOrth(Yf_Minus, [WS.Up; WS.Uii; WS.Yp; WS.Yii; Uf_Minus]);
        
        % Take SVD of YHatObUfRes
        [WS.YHatObUfRes_U,WS.YHatObUfRes_S,~] = svd(WS.YHatObUfRes, 'econ');
        
        clear Yf Yf_Minus Uf_Minus YHatOb  % Free up memory
    end
    
    S2 = WS.YHatObUfRes_S(1:n2, 1:n2);
    U2 = WS.YHatObUfRes_U( :  , 1:n2);

    Oy = U2 * S2^(1/2);                          
    Oy_Minus = Oy(1:(end-ny), :);               
    
    Xk2 = pinv(Oy) * WS.YHat;  % Eq.(28)
    Xk2_Plus1 = pinv(Oy_Minus) * WS.YHatMinus;   
    
    Xk = cat(1, Xk, Xk2);
    Xk_Plus1 = cat(1, Xk_Plus1, Xk2_Plus1);
    clear Xk2 Xk2_Plus1 % Free up mem
end

% Parameter identification
% ------------------------
idSys = learnParamsFromStates(y, z, u, nx, n1, k, WS, n2, Xk, Xk_Plus1, nz, ny, nu, N, fit_Cz_via_KF);

idSys.YPrepModel = YPrepModel;
idSys.ZPrepModel = ZPrepModel;
idSys.UPrepModel = UPrepModel;
    
end

function idSys = learnParamsFromStates(y, z, u, nx, n1, k, WS, n2, Xk, Xk_Plus1, nz, ny, nu, N, fit_Cz_via_KF)
    % Fit all model parameters associated with x1 and x2 (optional) given
    % their estimates Xk, Xk_Plus1 at current time and the next time step. 
    % These estimates are obtained via subspace identification.
    if n1 > 0
        % A associated with the z-related states
        [XkP1Hat, A1Tmp] = projOrth( Xk_Plus1(1:n1, :), [Xk(1:n1, :); WS.Uf] );  
        A = A1Tmp(1:n1, 1:n1);
        w = Xk_Plus1(1:n1, :) - XkP1Hat(1:n1, :);    % VODM Book, (4.51) term 3
        clear XkP1Hat % Free up memory
    else
        A = [];
        w = [];
    end

    if n2 > 0
        % A associated with the other states (x2)
        [XkP1Hat, A23Tmp] = projOrth( Xk_Plus1((n1+1):end, :), [Xk(:, :); WS.Uf] ); 
        A23 = A23Tmp(:, 1:nx);
        A = [A, zeros(n1, n2); A23];
        w = cat(1, w, Xk_Plus1((n1+1):end, :) - XkP1Hat); 
        clear XkP1Hat % Free up memory
    end

    if nz > 0
        [ZiiHat, CzTmp] = projOrth( WS.Zii, [Xk; WS.Uf] );
        Cz = CzTmp(:, 1:nx);
        e = WS.Zii            - ZiiHat;                         
    else
        Cz = [];
    end

    [YiiHat, CyTmp] = projOrth( WS.Yii, [Xk; WS.Uf] );
    Cy = CyTmp(:, 1:nx);
    v = WS.Yii - YiiHat;   % Eq.(35)                                     

    clear Xk Xk_Plus1 YiiHat % Free up memory

    % Compute noise covariances
    Q = (w * w.')/N; % Eq.(36)
    S = (w * v.')/N; % Eq.(36)
    R = (v * v.')/N; % Eq.(36)

    Q = (Q + Q.')/2; % Make precisely symmetric
    R = (R + R.')/2; % Make precisely symmetric

    if nz > 0
        Sxz = (w * e.')/N;
        Syz = (v * e.')/N;
        Rz  = (e * e.')/N;
        Rz  = (Rz + Rz.')/2; % Make precisely symmetric
    else
        Sxz = [];
        Syz = [];
        Rz  = [];
    end

    % Compute secondary system matrices, e.g. Kalman gain, etc
    try
        [P,~,GG] = dare(A',Cy',Q,R,S); 
        K = GG'; 
        innovCov = Cy*P*Cy.' + R;

    catch ME
        fprintf('WARNING: %s\n', ME.message);
        P = nan(size(A, 1));
        K = nan(size(A, 1), size(Cy, 1));
        innovCov = nan(size(Cy, 1));
    end

    if all(abs(eig(A)) < 1) % Stable
        xCov = dlyap(A,Q);
        G = A*xCov*Cy' + S;
        YCov = Cy*xCov*Cy' + R; % Covariance matrix of output
        xPCov = xCov - P;       % Covariance of Kalamn predicted states
    else
        xCov = nan(size(A, 1));
        G = nan(size(A, 1), size(Cy, 1));
        YCov = nan(size(Cy, 1));
        xPCov = nan(size(A, 1));
    end

    if nu > 0 % Following a procedure similar to VODM Book, pages 125-127 to find the least squares solution for the model parameters B and Dy
        RR = triu(qr([WS.Up; WS.Uf; WS.Yp; WS.Yf].'/sqrt(N))).';
        RR = RR(1:((2*nu+2*ny)*k), 1:((2*nu+2*ny)*k));

        RUf = RR((nu*k)+(1:(k*nu)), :);
        RYf = RR(((nu*2+ny)*k)+(1:(k*ny)), :);
        RYf_Minus = RR(((nu*2+ny)*k+ny)+(1:((k-1)*ny)), :);
        RYii = RR(((nu*2+ny)*k)+(1:ny), :);

        YHat = [RYf(:, 1:((2*nu+ny)*k)), zeros(k*ny,ny)];
        YHatMinus = RYf_Minus(:, 1:((2*nu+ny)*k+ny));
        Yii = RYii(:, 1:((2*nu+ny)*k+ny));
        Uf = RUf(:, 1:((2*nu+ny)*k+ny));

        % Recompute Oy and Oy_Minus using A and Cy and recompute Xk and Xk_Plus1 using the new Oy
        [Xk, Xk_Plus1] = recomputeObsAndStates(A, Cy, k, YHat, YHatMinus);
        [B, Dy] = computeBD( A, Cy, Yii, Xk_Plus1, Xk, k, nu, Uf);
        if nz > 0
            % We can fit Dz via a regression from the extracted Kalman states
            idSysY = struct('A', A, 'B', B, 'C', Cy, 'D', Dy, 'Q', Q, 'R', R, 'S', S, 'K', K, 'Cz', Cz);
            [~, ~, xPred] = PSIDPredict(idSysY, y.', u.'); % Eq. (37)

            if fit_Cz_via_KF
                CzDz = z / [xPred.'; u];
                Cz = CzDz(:,1:nx);
                Dz = CzDz(:,nx+1:end);
            else
                Dz = (z - Cz * xPred.') / u; % Eq.(38)
            end

        else
            Dz = [];
        end
        clear Xk Xk_Plus1 % Free up memory
    else
        B = [];
        Dy = [];
        Dz = [];
    end

    idSys = struct( ...
        'A', A, ...
        'B', B, ...
        'Cy', Cy, ...
        'Cz', Cz, ...
        'Dy', Dy, ...
        'Dz', Dz, ...
        'Q', Q, ...
        'R', R, ...
        'S', S, ...
        'Rz', Rz, ...
        'Sxz', Sxz, ...
        'Syz', Syz, ...
        'G', G, ...
        'YCov', YCov, ...
        'K', K, ...
        'innovCov', innovCov, ...
        'P', P, ...
        'xPCov', xPCov, ...
        'xCov', xCov, ...
        'n1', n1 ...
     );
end

function A_AHat = removeProjOrth(A, B) % A/B
    % Returns the part of A that is not linearly predictable using B
    A_AHat = A - projOrth(A, B);
end

function [AHat, W] = projOrth(A, B) % A/B
    % Returns the best linear prediction of A using B
    % Decomposes A into a linear combination of two orthogonal matrices (B and B_orth)
    % Projects the row space of A orthogonally onto the row space of B
    if ~isempty(B)
        W = (A / B);
        AHat = W * B; % or: A * B.' * pinv(B * B.') * B;
    else
        W = zeros(size(A, 1), size(B, 1));
        AHat = zeros(size(A));
    end
end

function [AHat, W] = projOblique(A, B, C) % A_{C}/B 
    % Returns best prediction of A using B, for the parts of A that are not predictable using C
    % Decomposes A into a linear combination of two non-orthogonal matrices (B and C)
    % Projects the row space of A along the row space of C onto the row space of B
	if ~isempty(C)
        A_C = A - projOrth(A, C); 
        B_C = B - projOrth(B, C); 
        W = (A_C / B_C);
        AHat = W * B; 
	else
        [AHat, W] = projOrth(A, B); 
	end
end

function [Xk, Xk_Plus1] = recomputeObsAndStates(A, C, k, YHat, YHatMinus)
    % Compute observabilioty matrices Oy and Oy_Minus using A and C and
    % recompute Xk and Xk_Plus1 using the new Oy and Oy_Minus
    % Outputs:
    % 1) Xk: recomputed states 
    % 2) Xk_Plus1: recomputed states at next time step
    [Oy, Oy_Minus] = computeObsFromAC(A, C, k);

    Xk = pinv(Oy) * YHat;                    
    Xk_Plus1 = pinv(Oy_Minus) * YHatMinus;   
end

function [B, D] = computeBD( A, C, Yii, Xk_Plus1, Xk, k, nu, Uf)
    % Compute matrices corresponding to the effect of external input
    % Outputs:
    % 1)B and 2)D matrices in the following state space equations
    % x(k) = A * x(k) + B * u(k) + w(k)
    % y(k) = Cy * x(k) + Dy * u(k) + v(k)
    
    % Find B and D
    [Oy, Oy_Minus] = computeObsFromAC(A, C, k);
    
    % See ref.40, pages 125-127
    PP = [Xk_Plus1 - A * Xk;
          Yii      - C * Xk];

    L1 = A * pinv(Oy);
    L2 = C * pinv(Oy);
    
    nx = size(A, 1);
    ny = size(C, 1);
    
    ZM = [zeros(nx,ny) pinv( Oy_Minus )];
    
    % LHS * DB = PP    
    LHS = zeros( numel(PP), (nx+ny)*nu );
    RMul = blkdiag( eye(ny), Oy_Minus );
    
    NN = cell(k, 1); % ref.40 (4.54), (4.57) .. ,(4.59)
    % Plug in the terms into NN
    for i = 1:k
        NN{i} = zeros( (nx+ny), k*ny );   
        
        NN{i}(    1:nx , 1:((k-i+1)*ny)) =  ZM( :, ((i-1)*ny+1):end ) ...
                                          - L1( :, ((i-1)*ny+1):end );
        NN{i}(nx+(1:ny), 1:((k-i+1)*ny)) = -L2( :, ((i-1)*ny+1):end );
        if i == 1
            NN{i}(nx+(1:ny), 1:ny) = NN{i}(nx+(1:ny), 1:ny) + eye(ny);
        end
        
        % Plug into LHS
        LHS = LHS + kron( Uf((i-1)*nu + (1:nu), :).', NN{i} * RMul);
    end
    
    DBVec = LHS \ PP(:);
    DB = reshape(DBVec, [nx+ny, nu]);
    D = DB(1:ny, :);
    B = DB(ny+(1:nx), :);
end

function [P, K, innovCov, xCov, G, YCov, xPCov] = computeKalmanParams(A, Cy, Q, R, S)
    % Compute secondary model parameters essential for running Kalman filter
    try
        [P,~,GG] = dare(A',Cy',Q,R,S); 
        K = GG'; 
        innovCov = Cy*P*Cy.' + R;

    catch ME
        fprintf('WARNING: %s\n', ME.message);
        P = nan(size(A, 1));
        K = nan(size(A, 1), size(Cy, 1));
        innovCov = nan(size(Cy, 1));
    end

    if all(abs(eig(A)) < 1) % Stable
        xCov = dlyap(A,Q);
        G = A*xCov*Cy' + S;
        YCov = Cy*xCov*Cy' + R; % Covariance matrix of output
        xPCov = xCov - P;       % Covariance of Kalamn predicted states
    else
        xCov = nan(size(A, 1));
        G = nan(size(A, 1), size(Cy, 1));
        YCov = nan(size(Cy, 1));
        xPCov = nan(size(A, 1));
    end

end
