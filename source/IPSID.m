% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IPSID: Input Preferential Subspace Identification Algorithm
% Publication: P. Vahidi*, O. G. Sani*, and M. M. Shanechi, "Modeling and dissociation of 
%              intrinsic and input-driven neural population dynamics underlying behavior", PNAS (2024). 
% * Comments within the documentation that refer to Eq. (XX), Figures, and Notes are referencing the above paper.
% IPSID identifies a linear stochastic model for a signal y, while prioritizing
% the latent states that are predictive of another signal z, while a known
% external input u is applied to the system.
% The complete model is as follows:
% [x1(k+1); x2(k+1); x3(k+1)] = [A11 0 0; A21 A22 0;0 0 A33] * [x1(k); x2(k); x3(k)] + [B1; B2; B3] * u(k) + w(k)
%                        y(k) = [Cy1   Cy2      0] * [x1(k); x2(k); x3(k)] + [Dy] * u(k) + v(k)
%                        z(k) = [Cz1     0    Cz3] * [x1(k); x2(k); x3(k)] + [Dz] * u(k) + e(k)
% x(k) = [x1(k); x2(k); x3(k)] => Latent state time series
% x1(k) => Latent states related to z ( the pair (A11, Cz1) is observable )
% x2(k) => Latent states unrelated to z 
% x3(k) => Latent states related to z but unrelated to y
% u(k) => External input that was applied to the system
% Given training time series from y(k), z(k) and u(k), the dimension of x(k) 
% (i.e. nx), and the dimension of x1(k) (i.e. n1), and the dimension of x3(k) (i.e. n3) the algorithm finds 
% all model parameters and noise statistics:
%       - A  : [A11 0 0; A21 A22 0;0 0 A33]
%       - Cy : [Cy1   Cy2      0]
%       - Cz : [Cz1     0    Cz3]
%       - B  : [B1     B2     B3]
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
%     - (9)  time_first (default=false): Only the false option supported for now,
%               if true, will expect the time dimension of the data to be the 
%               first dimension (e.g. Z is T x nz). If false, will expect time to
%               be the second dimension in all data (e.g. Z is nz x T).
%     - (10) remove_nonYrelated_fromX1 (default=false): If remove_nonYrelated_fromX1=true, 
%               the direct effect of input u(k) on z(k) would be excluded from x1(k) in 
%               additional step 1 (preprocessing stage). If false, additional step 1 won't 
%               happen and x3 (and its corresponding model parameters [A33, B3, Cz3 and noise 
%               statistics related to x3]) won't be learned even if n3>0 provided.
%     - (11) n_pre (default=inf): preprocessing dimension used in additional step 1. 
%               Additional step 1 only happens if remove_nonYrelated_fromX1=true. 
%               Large values of n_pre (assuming there is enough data to fit models with 
%               such large state dimensions) would ensure all dynamics of Y are preserved in
%               the preprocessing step. 
%               If, n_pre=np.inf, n_pre will be automatically set to the largest possible value given the data 
%               (all available SVD dimensions).
%               If n_pre=0, Additional steps 1 and 2 won't happen and x3 won't be learned 
%               (remove_nonYrelated_fromX1 will be set to false, n3 will be 0).
%     - (12) n3: number of latent states x3(k) in the optional additional step 2.
%     - (13) remove_mean_Y: if true will remove the mean of y. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (14) remove_mean_Z: if true will remove the mean of z. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (15) remove_mean_U: if true will remove the mean of u. 
%               Must be true if data is not zero mean. Defaults to true.
%     - (16) zscore_Y: if true will z-score y. It is ok to set this to false,
%                     but setting to true may help with stopping some dimensions of 
%                     data from dominating others. Defaults to false.
%     - (17) zscore_Z: if true will z-score Z. It is ok to set this to false,
%                     but setting to True may help with stopping some dimensions of 
%                     data from dominating others. Defaults to true.
%     - (18) zscore_U: if true will z-score U. It is ok to set this to false,
%                     but setting to True may help with stopping some dimensions of 
%                     data from dominating others. Defaults to true.
%   Outputs:
%     - (1) idSys: structure with the system parameters for the identified
%           system. Will have the following fields (defined above):
%               'A', 'Cy', 'Cz', 'Q', 'R', 'S'
%               'B', 'Dy', 'Dz'
%               'G', 'YCov', 'K', 'innovCov', 'P', 'xPCov', 'xCov' 
% 
%     Notes:
%         (1) Preprocessing step (refer to Fig. S5 - top row, and Note S2) is optional and won't happen by default. To enable, provide remove_nonYrelated_fromX1=true, n_pre>0.
%         (2) In case preprocessing step enabled (see Note 1), Parameter Dz won't be fitted (forced to 0) and full Cz is the least square solution for estimating z given  full x.
%         (3) Learning x3 and fitting its corresponding parameters are optional and won't happen by default. To enable, provide n3>0, and enable preprocessing step (see Note 1).
%         (4) If the parameter n_pre>0 and remove_nonYrelated_fromX1=true , the "direct effect of input u(k) on z(k)" would be excluded from x1(k) in a preprocessing stage.
%             In this case, the preprocessing dimension (i.e. n_pre) determines the preprocessing dimension.
%         (5) PSID (Preferential Subspace Identification) can be performed as a special case using IPSID algorithm. To do so, simply set u=[].
%         (6) INDM (or ISID, i.e., Subspace Identification with input U, unsupervised by Z) can be performed as a special case of IPSID. To do so, simply set z=[], n1=0.
%         (7) NMDM (or SID, i.e., Standard Subspace Identification without input U, unsupervised by Z) can be performed as a special case of IPSID. To do so, simply set z=[], u=[], n1=0.
% 
%   Usage example:
%       idSys = IPSID(y, z, u, nx, n1, k);
%       idSys = IPSID(y, z, u, nx, n1, k, 'remove_nonYrelated_inX1', true, 'n_pre', n_pre);  % With external input and preprocessing x1. (n_pre>0)
%       idSys = IPSID(Y, Z, u, nx, n1, k, 'remove_nonYrelated_inX1', true, 'n_pre', n_pre, 'n3', n3);  % With external input, preprocessing x1 and optional states x3. (n_pre>0, n3>0)
%       [idSys, WS] = IPSID(y, z, u, nx, n1, k, 'WS',WS); 
%       idSysISID = IPSID(y, z, u, nx, 0, k); % Set n1=0 for ISID
%       idSysPSID = IPSID(y, z, [], nx, n1, k); % Set u=[] for PSID
%       idSysSID = IPSID(y, z, [], nx, 0, k); % Set n1=0 and u=[] for SID

function [idSys, WS] = IPSID(y, z, u, nx, n1, k, varargin)

    p = inputParser;
    
    addParameter(p, 'WS', struct, @isstruct);
    addParameter(p, 'fit_Cz_via_KF', true, @islogical);
    addParameter(p, 'time_first', false, @islogical);
    addParameter(p, 'remove_nonYrelated_fromX1', false, @islogical);
    addParameter(p, 'n_pre', inf, @isnumeric);
    addParameter(p, 'n3', 0, @isnumeric);
    
    addParameter(p, 'remove_mean_Y', true, @islogical);
    addParameter(p, 'remove_mean_Z', true, @islogical);
    addParameter(p, 'remove_mean_U', true, @islogical);
    addParameter(p, 'zscore_Y', false, @islogical);
    addParameter(p, 'zscore_Z', false, @islogical);
    addParameter(p, 'zscore_U', false, @islogical);
    
    parse(p,varargin{:});
    
    WS = p.Results.WS;
    time_first = p.Results.time_first;
    fit_Cz_via_KF = p.Results.fit_Cz_via_KF;
    remove_nonYrelated_fromX1 = p.Results.remove_nonYrelated_fromX1;
    n_pre = p.Results.n_pre;
    n3 = p.Results.n3;
    
    if time_first, error('Time as the first dimension is not supported for IPSID yet. Provide data in: data_dim*time'); end
    if iscell(y), error('Trial-based learning by IPSID is not suported yet. Please use the Python implementation available at: https://github.com/ShanechiLab/PyPSID'); end
    
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
    
    if nu==0 || n1==0 % Since the external input u and/or n1 is not provided, preprocessing step is disabled and x3 won't be learned.
        remove_nonYrelated_fromX1 = false;
        n_pre = 0; 
        n3 = 0;
    end
    
    if ~remove_nonYrelated_fromX1 || n_pre==0 % Due to provided settings, preprocessing step is disabled and x3 won't be learned.
        remove_nonYrelated_fromX1 = false;
        n_pre = 0;
        n3 = 0;
    end
    
    
    if n1 > 0 && nz > 0
        if ~isfield(WS, 'ZHatObUfRes_U') || isempty(WS.ZHatObUfRes_U) 
            WS.Zf = blkhankskip(z, k, N, k, time_first);
            %######### Additional step1/Preprocessing ((Vahidi, Sani et al) Fig. S5, top row) ##########
            if remove_nonYrelated_fromX1 % preprocessing step (Fig. S5, top row)
                Yf_Minus = WS.Yf((ny+1):end, :); 
                Uf_Minus = WS.Uf((nu+1):end, :);
                YHatOb_pr = projOblique(WS.Yf, [WS.Up; WS.Yp], WS.Uf);
                YHatObRes_pr = removeProjOrth(YHatOb_pr, WS.Uf);      
                [U0, S0, ~] = svd(YHatObRes_pr, 'econ');            
                if n_pre <= size(U0, 2)
                    keepDims = n_pre;
                else
                    keepDims = size(U0, 2); 
                end
                U0 = U0(:,1:keepDims);
                S0 = S0(1:keepDims,1:keepDims); 
                YObs = U0 * S0.^(1/2);
                Xk_pre = pinv(YObs) * projOrth(WS.Yf, [WS.Up; WS.Yp; WS.Uf]); 
                [ZPr, Qz] = projOblique(WS.Zf, Xk_pre, [WS.Up; WS.Uf]); % Eq.(39)
                Zf = ZPr;
                
                YObs_ = YObs(1:(end-ny), :); 
                XkMinus_pre = pinv(YObs_) * projOrth(Yf_Minus, [WS.Up; WS.Uii; WS.Yp; WS.Yii; Uf_Minus]);        
                Qz_Minus1 = Qz(1:(end-nz),:);
                ZPr = Qz_Minus1 * XkMinus_pre;
                Zf_Minus = ZPr; 
            %###################################################
            else
                Zf = WS.Zf;
                Zf_Minus = Zf((nz+1):end, :);
            end
            Uf_Minus = WS.Uf((nu+1):end, :);
               
            % IPSID Stage 1:
            % -----------------------
            % Oblique projection of Zf along Uf, onto [Up;Yp]: Eq.(22) 
            ZHatOb = projOblique(Zf, [WS.Up; WS.Yp], WS.Uf); 
            WS.ZHatObUfRes = removeProjOrth(ZHatOb, WS.Uf);
            
            % Orthogonal projection of Zf onto [Up;Yp;Uf]
            WS.ZHat = projOrth(Zf, [WS.Up; WS.Yp; WS.Uf]);
    
            % Orthogonal projection of Zf_Minus onto [Up_plus;Yp_plus;Uf_Minus]
            WS.ZHatMinus = projOrth(Zf_Minus, [WS.Up; WS.Uii; WS.Yp; WS.Yii; Uf_Minus]);
    
            % Take SVD of ZHatObUfRes
            [WS.ZHatObUfRes_U,WS.ZHatObUfRes_S,~] = svd(WS.ZHatObUfRes, 'econ'); % Eq. (23)
            
            clear Zf Zf_Minus Uf_Minus ZHatOb % Free up memory
        end
    
        Sz = WS.ZHatObUfRes_S(1:n1, 1:n1); 
        Uz = WS.ZHatObUfRes_U( :  , 1:n1);
        
        Oz = Uz * Sz^(1/2);                         
        Oz_Minus = Oz(1:(end-nz), :);               
        
        Xk = pinv(Oz) * WS.ZHat;                    % Eq.(24)
        Xk_Plus1 = pinv(Oz_Minus) * WS.ZHatMinus;   
       
    else
        n1 = 0;
        Xk = [];
        Xk_Plus1 = [];
    end
    
    
    n2 = nx - n1;
    if n3 > 0 % In case asked to dedicate some model capacity (state dimension) to X3, then recompute dimension of X21
        n2 = max(0, nx - n1 - n3); % Anything remaining from nx after allocating n1 and n3 becomes n2
        n3 = nx - n1 - n2; % The dimension of final model would be equal to final n1+n2+n3 based on their adjusted values (which is equal to the input nx).
        nx = n1 + n2; % This is the nx used in 2-stage IPSID algorithm (without considering X3) i.e., dim([X1;X2])
    end
    
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
                Yf = Yf - YHatOb1; % Eq.(25)
                clear YHatOb1
                
                Oy1_Minus = Oy1(1:(end-ny), :);     
                Yf_Minus = Yf_Minus - Oy1_Minus * Xk_Plus1;  
            end
            
            % Oblique projection of Yf along Uf, onto [Up;Yp]: Eq.(26)
            YHatOb = projOblique(Yf, [WS.Up; WS.Yp], WS.Uf); 
            WS.YHatObUfRes = removeProjOrth(YHatOb, WS.Uf);
            
            % Orthogonal projection of Yf onto [Up;Yp;Uf]
            WS.YHat = projOrth(Yf, [WS.Up; WS.Yp; WS.Uf]);
    
            % Orthogonal projection of Yf_Minus onto[Up_plus;Yp_plus;Uf_Minus]
            WS.YHatMinus = projOrth(Yf_Minus, [WS.Up; WS.Uii; WS.Yp; WS.Yii; Uf_Minus]);
            
            % Take SVD of YHatObUfRes
            [WS.YHatObUfRes_U,WS.YHatObUfRes_S,~] = svd(WS.YHatObUfRes, 'econ'); % Eq.(27)
            
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
    
    % Parameter identification excluding x3 (if any)
    % ------------------------
    idSys = learnParamsFromStates(y, z, u, nx, n1, k, WS, n2, Xk, Xk_Plus1, nz, ny, nu, N, remove_nonYrelated_fromX1, fit_Cz_via_KF);
    
    % Additional step 2/Learning x3 and its model parameters (if desired): Fig. S5 bottom row, Note S2
    % ------------------------
    if n3 > 0
        % Learn n3 additional stated that optimize the forward prediction from u to the residual of z
        idSysKF = struct('a', idSys.A, 'b', idSys.B, 'c', idSys.Cy, 'd', idSys.Dy, 'q', idSys.Q, 'r', idSys.R, 's', idSys.S, 'k', idSys.K, 'Cz', idSys.Cz);
        [~, ~, x12Pred] = PSIDPredict(idSysKF, y.', u.');
        zRes = z - idSys.Cz * x12Pred' - idSys.Dz * u;
        
        % Using Stage 2 of IPSID alone for identifying dynamics in residual z (zRes) driven by u
        idSysRes = IPSID(double(zRes), [], double(u), n3, 0, k); 
        idSysRes_ = struct('A', idSysRes.A, 'B', idSysRes.B, 'Cz', idSysRes.Cy, 'Dz', idSysRes.Dy, 'Q', zeros(size(idSysRes.A)), 'R', zeros(size(idSys.Cy,1),size(idSys.Cy,1)), 'S', zeros(size(idSysRes.A,1),size(idSys.Cy,1)));
        %  ---------------------------------------------
    
        idSys = combineIdSysWithEps(idSys, idSysRes_); % Combining model parametrs learned for [x1x2] and [x3] in a single model
    end
    
    idSys.YPrepModel = YPrepModel;
    idSys.ZPrepModel = ZPrepModel;
    idSys.UPrepModel = UPrepModel;
        
    end
    
    function idSys = learnParamsFromStates(y, z, u, nx, n1, k, WS, n2, Xk, Xk_Plus1, nz, ny, nu, N, remove_nonYrelated_fromX1, fit_Cz_via_KF)
        % Fit all model parameters associated with x1 and x2 (optional) given
        % their estimates Xk, Xk_Plus1 at current time and the next time step. 
        % These estimates are obtained via subspace identification.
        if n1 > 0
            % A associated with the z-related states
            [XkP1Hat, A1Tmp] = projOrth( Xk_Plus1(1:n1, :), [Xk(1:n1, :); WS.Uf] );  % Eq.(29)
            A = A1Tmp(1:n1, 1:n1);
            w = Xk_Plus1(1:n1, :) - XkP1Hat(1:n1, :);    % Eq.(33)           % VODM Book, (4.51) term 3
            clear XkP1Hat % Free up memory
        else
            A = [];
            w = [];
        end
    
        if n2 > 0
            % A associated with the other states (x2)
            [XkP1Hat, A23Tmp] = projOrth( Xk_Plus1((n1+1):end, :), [Xk(:, :); WS.Uf] );  % Eq.(30)
            A23 = A23Tmp(:, 1:nx);
            A = [A, zeros(n1, n2); A23];
            w = cat(1, w, Xk_Plus1((n1+1):end, :) - XkP1Hat);   % Eq.(34)    
            clear XkP1Hat % Free up memory
        end
    
        if nz > 0
            [ZiiHat, CzTmp] = projOrth( WS.Zii, [Xk; WS.Uf] );      % Eq.(32)
            Cz = CzTmp(:, 1:nx);
            e = WS.Zii            - ZiiHat;                         
        else
            Cz = [];
        end
    
        [YiiHat, CyTmp] = projOrth( WS.Yii, [Xk; WS.Uf] );   % Eq.(31)        
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
    
        if nu > 0 % Following a procedure similar to ref.40, pages 125-127 to find the least squares solution for the model parameters B and Dy
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
    
                if ~remove_nonYrelated_fromX1
                    if fit_Cz_via_KF
                        CzDz = z / [xPred.'; u];
                        Cz = CzDz(:,1:nx);
                        Dz = CzDz(:,nx+1:end);
                    else
                        Dz = (z - Cz * xPred.') / u; % Eq.(38)
                    end
                else
                    Cz = z / xPred.'; % Eq.(40) Fitting Z-readout from all states in case of using additional steps (preprocessing)
                    Dz = zeros(nz, nu); % Enforcing no feedthrough to z in case of using additional steps (preprocessing)
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
    
    function idSysNew = combineIdSysWithEps(idSys, idSysZN)
        % Creates and outputs a single model by combining parameters of:
        % s: Main model, parameters associated with X1, X2 in IPSID stage 1, 2
        % s3: Optional model, parameters associated with X3 in IPSID additional step 2
        idSysNew = idSys;
        idSysNew.A = blkdiag(idSys.A, idSysZN.A);
        idSysNew.B = [idSys.B; idSysZN.B];
        idSysNew.Cy = [idSys.Cy, zeros(size(idSys.Cy,1), size(idSysZN.A, 1))];
        if isfield(idSys, 'Cz') && ~isempty(idSys.Cz) && isfield(idSysZN, 'Cz')  && ~isempty(idSysZN.Cz)
            idSysNew.Cz = [idSys.Cz, idSysZN.Cz];
        elseif  isfield(idSysZN, 'Cz')  && ~isempty(idSysZN.Cz)
            idSysNew.Cz = idSysZN.Cz;
        end
        if isfield(idSys, 'Dz') && ~isempty(idSys.Dz) && isfield(idSysZN, 'Dz') && ~isempty(idSysZN.Dz)
            idSysNew.Dz = idSys.Dz + idSysZN.Dz;
        elseif  isfield(idSysZN, 'Dz') && ~isempty(idSysZN.Dz)
            idSysNew.Dz = idSysZN.Dz;
        end
        
        idSysNew.Q = blkdiag(idSys.Q, idSysZN.Q);
        idSysNew.S = [idSys.S; 0 * idSysZN.S];
        % Note: Sxz, Syz, and Rz are never used in inference so we set them to nan for now, 
        % but technically should be updated here after adding X3 to the model
        idSysNew.Sxz = nan * [idSys.Sxz; zeros(size(idSysZN.A,1), size(idSys.Cz,1))];
        idSysNew.Syz = nan * idSysNew.Syz;
        idSysNew.Rz = nan * idSysNew.Rz;
        
        [idSysNew.P, idSysNew.K, idSysNew.innovCov, idSysNew.xCov, idSysNew.G, idSysNew.YCov, idSysNew.xPCov] ...
             = computeKalmanParams(idSysNew.A, idSysNew.Cy, idSysNew.Q, idSysNew.R, idSysNew.S);
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
    