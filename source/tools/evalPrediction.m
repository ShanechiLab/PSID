% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%evalPrediction Evaluates prediction performance with the given measure
%   Inputs:
%     - (1) trueVals: True values. Rows are samples and columns are
%                     different sets of data. Each column is compared
%                     separately with the corresponding column in the
%                     prediction matrix and the performance measure is
%                     calculated for it.
%     - (2) predictedVals: Predicted values (must have same dimensions as trueVals)
%     - (3) measure: performance measure. Can be any of the following:
%                    - 'CC': correlation coefficient.
%                    - 'R2': R-squared (aka coefficient of determination)
%                    - 'RMSE': Root mean squared error
%   Outputs:
%     - (1) out: the performance measure calculated for each column of the
%     input.

function out = evalPrediction(trueVals, predictedVals, measure)

if any(size(trueVals)~=size(predictedVals)), error('Size mismatch between true and predicted values!\n'); end

err = predictedVals - trueVals;

if strcmpi(measure, 'CC')
    [rho, pVal] = corr(trueVals, predictedVals);
    out = diag(rho).';
elseif strcmpi(measure, 'R2')
    out = ones(1, size(trueVals, 2)) - mean(err.^2, 1) ./ mean((trueVals-repmat(mean(trueVals, 1), [size(trueVals, 1) 1, 1])).^2);
elseif strcmpi(measure, 'MSE')
    out = mean(err.^2, 1);
elseif strcmpi(measure, 'RMSE')
    out = sqrt(mean(err.^2, 1));
else
    error('Performance measure "%s" is not supported!\n', measure);
end

end