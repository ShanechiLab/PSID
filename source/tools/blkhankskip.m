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
%     - (1) y: Inputs time series. Must be N x ny (unless time_first=false):
%               [y(1), y(2), y(3), ..., y(N)]
%               If input is a cell array, will call blkhankskip for each
%               cell and concatenate the results.
%     - (2) k: number of block rows
%     - (3) j (default: N - i + 1 ): number of columns
%     - (4) s (default: 0): skip the first s samples of y
%     - (5) time_first (default: true): if true, will expect the time dimension 
%             of the data to be the first dimension (e.g. y is T x ny). If false, 
%             will expect time to be the second dimension in all data 
%             (e.g. y is ny x T).
%   Output:
%     - (1) H: The constructed block Hankel matrix
% 

function H = blkhankskip(y, k, j, s, time_first)

if nargin < 5, time_first = true; end

if iscell(y) % If cell array, extract hankel matrices from each element
    if nargin < 3 || isempty(j), j = cell(size(y)); end
    if nargin < 4 || isempty(s), s = 0; end
    
    H = [];
    for yInd = 1:numel(y)
        thisH = blkhankskip(y{yInd}, k, j(yInd), s, time_first);
        H = cat(2, H, thisH);
    end
    return
end

[ny, N] = getHSize(y, 0, time_first);

if nargin < 3 || isempty(j), j = N - 2*k + 1; end
if nargin < 4 || isempty(s), s = 0; end

H = nan( ny * k, j );
for r = 1:k
    if time_first
        thisBlock = y( (s+r-1) + (1:j), : )';
    else
        thisBlock = y( :, (s+r-1) + (1:j) );
    end
    H( (r-1)*ny + (1:ny), : ) = thisBlock;
end

end
