% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% projOrth Projects A onto B. A and B must be wide matrices with dim x samples.
%   Inputs:
%     - (1) A: data matix 1, must be n_a x N
%     - (2) B: data matix 1, must be n_b x N
%   Output:
%     - (1) AHat: projection of A onto B
%     - (2) W: The matrix that gives AHat when it is right multiplied by B
    
function [AHat, W] = projOrth(A, B)

W = A / B;      % or: A / B = A * B.' * pinv(B * B.') 
AHat = W * B;   % or: A * B.' * pinv(B * B.') * B

end
