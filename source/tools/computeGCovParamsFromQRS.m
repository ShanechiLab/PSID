% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computeGCovParamsFromQRS Computes the secondary system parameters G, YCov
%   Inputs:
%     - (1) A: State transfer matrix
%     - (2) C: Observation matrix
%     - (3) Q: state noise covariance
%     - (4) R: observation noise covariance
%     - (5) S: state-observation noise cross covariance
%     - (6) xCov (optional): state covariance (solution to dlyap(A,Q))
%   Outputs:
%     - (1) G: Cov( x(k+1), y(k) )
%     - (2) Cov( y(k), y(k) )
%     - (3) xCov: state covariance (solution to dlyap(A,Q))
%   Usage example:
%       [G, YCov, xCov] = computeGCovParamsFromQRS( A, C, Q, R, S )

function [G, YCov, xCov] = computeGCovParamsFromQRS( A, C, Q, R, S, xCov )

if nargin < 6, xCov = []; end
if all(abs(eig(A)) < 1) % Stable
    if isempty(xCov)
        xCov = dlyap(A,Q);
    end
    G = A*xCov*C' + S;
    YCov = C*xCov*C' + R; % Covariance matrix of output
else
    if isempty(xCov)
        xCov = nan(size(A, 1));
    end
    G = nan(size(A, 1), size(C, 1));
    YCov = nan(size(C, 1));
end

end
