% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSIDPredict: Given a PSID model, predicts behavior z from neural data y
%   Inputs:
%     - (1) idSys: Identified model returned from running PSID
%     - (2) y: Observation signal (e.g. neural signal). 
%               Must be T x ny:
%               [y(1); y(2); y(3); ...; y(T)]
%     - (3) u: [to be determined]
%   Outputs:
%     - (1) zPred: predicted behavior z using the provided observation y
%               Has dimensions T x nz
%               [zPred(1); zPred(2); zPred(3); ...; zPred(T)]
%               with z(i) the best prediction of z(i) using y(1),...,y(i-1)
%     - (2) yPred (optional): same as (1), for the observation y itself
%     - (3) xPred (optional): same as (1), for the latent states
%   Usage example:
%       idSys = PSID(yTrain, zTrain, nx, n1, i);
%       [zPred, ~, xPred] = PSIDPredict(idSys, yTest);

function [zPred, yPred, xPred] = PSIDPredict(idSys, y, u, settings)

if nargin < 3, u = []; end
if nargin < 4, settings = struct; end

if iscell(y)
    zPred = cell(size(y));
    xPred = cell(size(y));
    yPred = cell(size(y));
    if isempty(u), u = cell(size(y)); end
    for yInd = 1:numel(y)
        [zPred{yInd}, yPred{yInd}, xPred{yInd}] = PSIDPredict(idSys, y{yInd}, u{yInd}, settings);
    end
    return
end

if ~isfield(idSys, 'Cz') || isempty(idSys.Cz) && isfield(idSys, 'T') && ~isempty(idSys.T) % For backwards compatibility
    idSys.Cz = idSys.T(2:end, :)';
end

% Run Kalman filter
N = size(y, 1);

A = fetchOneOfFieldValues(idSys, {'a', 'A'}, []);
K = fetchOneOfFieldValues(idSys, {'k', 'K'}, []);
Cy = fetchOneOfFieldValues(idSys, {'c', 'C', 'Cy'}, []);

nx = size(A, 1);
Xp = zeros(nx, 1); % Initial state
xPred = nan(N, nx);
for i = 1:N
    xPred(i, :) = Xp; % X(i|i-1)
    yThis = y(i, :);
    if isfield(idSys, 'YPrepModel') && ~isempty(idSys.YPrepModel)
        yThis = idSys.YPrepModel.apply(yThis); % Apply any mean-removal/zscoring
    end
    Xp = A * Xp  + K * (yThis' - Cy*Xp); % Kalman prediction
end

yPred = (Cy * xPred.').';
zPred = (idSys.Cz * xPred.').';
if isfield(idSys, 'YPrepModel') && ~isempty(idSys.YPrepModel)
    yPred = idSys.YPrepModel.apply_inverse(yPred); % Apply inverse of any mean-removal/zscoring
end
if isfield(idSys, 'ZPrepModel') && ~isempty(idSys.ZPrepModel) % Apply inverse of any mean-removal/zscoring
    zPred = idSys.ZPrepModel.apply_inverse(zPred);
end

end

function val = fetchOneOfFieldValues(csys, fNames, defaultVal)

val = defaultVal;

for fi = 1:numel(fNames)
    if isfield(csys, fNames{fi})
        val = csys.(fNames{fi});
        break
    end
end

end