% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% blkhankskip Forms a block Hankel matrix
% Given time series [y(1); y(2); y(3); ...; y(N)]
% An numbers i, j, and s. Forms the following block Hankel matrix:
% H = [ y(s+1),   y(s+2),   y(s+3), ...,   y(s+j  ) ;
%       y(s+2),   y(s+3),   y(s+4), ...,   y(s+j+1) ;
%       y(s+3),   y(s+4),   y(s+5), ...,   y(s+j+2) ;
%       ...
%       y(s+i), y(s+i+1), y(s+i+2), ..., y(s+j+i-1) ]
%   
%   Inputs:
%     - (1) y: Inputs time series (Must be ny x N):
%               [y(1), y(2), y(3), ..., y(N)]
%               If input is a cell array, will call blkhankskip for each
%               cell and concatenate the results.
%     - (2) k: number of block rows
%     - (3) j (default: N - i + 1 ): number of columns
%     - (4) s (default: 0): skip the first s samples of y
% 

function H = blkhankskip(y, k, j, s)

if iscell(y) % If cell array, extract hankel matrices from each element
    if nargin < 3, j = cell(size(y)); end
    if nargin < 4, s = 0; end
    
    H = [];
    for yInd = 1:numel(y)
        thisH = blkhankskip(y{yInd}, k, j(yInd), s);
        H = cat(2, H, thisH);
    end
    return
end

ny = size(y, 1); % Dimension of y
N  = size(y, 2); % Number of samples

if nargin < 3 || isempty(j), j = N - 2*k + 1; end
if nargin < 4 || isempty(s), s = 0; end

H = nan( ny * k, j );
for r = 1:k
    H( (r-1)*ny + (1:ny), : ) = y(:, (s+r-1) + (1:j) );
end

end
