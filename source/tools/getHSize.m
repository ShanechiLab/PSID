% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getHSize Extracts time and data dimension information and the expected size of 
% the block Hankel matrices that will be constructed using blkhankskip
%   Inputs:
%     - (1) y: Inputs time series. Must be N x ny (unless time_first=false):
%               [y(1), y(2), y(3), ..., y(N)]
%               If input is a cell array, will call blkhankskip for each
%               cell and concatenate the results.
%     - (2) i: number of block rows
%     - (3) time_first (default: true): if true, will expect the time dimension 
%             of the data to be the first dimension (e.g. y is T x ny). If false, 
%             will expect time to be the second dimension in all data 
%             (e.g. y is ny x T).
%   Output:
%     - (1) H: The constructed block Hankel matrix

function [ny, ySamples, N, y1] = getHSize(y, i, time_first)
    
ny = nan; y1 = nan; 
if ~iscell(y)
    if time_first
        [ySamples, ny] = size(y); 
    else
        [ny, ySamples] = size(y); 
    end
    N = ySamples - 2*i + 1;
    if ~isempty(y)
        y1 = y(1);
    end
else
    ySamples = [];
    N = [];
    for yInd = 1:numel(y)
        [nyThis, ySamplesThis, NThis, y1This] = getHSize(y{yInd}, i, time_first);
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
