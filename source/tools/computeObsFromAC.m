% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Omid Sani, Shanechi Lab, University of Southern California, 2019
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computeObsFromAC Computes the extended observability matrix from A and C
%   Inputs:
%     - (1) A: State transfer matrix
%     - (2) C: Observation matrix
%     - (3) k: horizon
%   Outputs:
%     - (1) Oy: extended observability matrix
%     - (2) Oy_Minus: extended observability matrix with the last block row
%                     removed
%   Usage example:
%       [Oy, Oy_Minus] = computeObsFromAC(A, C, k)

function [Oy, Oy_Minus] = computeObsFromAC(A, C, k)

ny = size(C, 1);
Oy = C;
for i = 1:(k-1)
    Oy = cat(1, Oy, Oy((i-1)*ny + (1:ny), :) * A);
end
if nargout > 1
    Oy_Minus = Oy(1:(end-ny), :);            
end
    
end
