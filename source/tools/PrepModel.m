% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PrepModel An object for keeping track of data preprocessing (mean removal and zscoring)
%   Usage example:
%       YPrepModel = PrepModel();
%       YPrepModel.fit(data);
%       YPrepModel.apply(data);

classdef PrepModel < handle & matlab.mixin.Copyable
    
    properties (SetAccess = private)
        dataMean        % mean of data
        dataStd         % std of data
        remove_mean     % bool specifying whether mean should be removed or not
        zscore          % bool specifying whether zscoring is needed or not
    end
    
    properties 
        
    end
    
    events
        
    end
    
    methods
        function obj = PrepModel(varargin)
            % Initializes a Preprocessing Model
            % Inputs: 
            % Any of the following can be provided as a key-value pair.
            % - mean: mean of data. Defaults to [].
            % - std: std of data. Defaults to [].
            % - remove_mean. If true, will remove the mean of data.
            %           Defaults to false.
            % - zscore. If true, will z-score the data.
            %           Defaults to false.
            % Outputs: 
            % - (none)
            p = inputParser;

            addParameter(p, 'mean', [], @isnumeric);
            addParameter(p, 'std', [], @isnumeric);
            addParameter(p, 'remove_mean', false, @islogical);
            addParameter(p, 'zscore', false, @islogical);

            parse(p,varargin{:});

            obj.dataMean = p.Results.mean;
            obj.dataStd = p.Results.std;
            obj.remove_mean = p.Results.remove_mean;
            obj.zscore = p.Results.zscore;
        end
        function fit(obj, Y, varargin)
            % Learns the preprocessing model from data
            % Inputs: 
            % (1) Y (numpy array or list of arrays): Input data. First 
            %           dimension must be time and the second dimension is 
            %           the data. Can be a cell array of data in which case 
            %           the stats will be learned from the concatenation of 
            %           all segments along the first dimension.
            % And any of the following can be provided as a key-value pair.
            % - remove_mean (bool, optional): If true, will remove the 
            %           mean of data. Defaults to True.
            % - zscore (bool, optional): If true, will zscore the data to 
            %           have unit std in all dimensions. Defaults to False.
            % - time_first (bool, optional): If true, will assume input 
            %           data has time as the first dimension. Otherwise 
            %           assumes time is the second dimension. In any case, 
            %           model will by default treat new data as if time is 
            %           the first dimension. Defaults to True.
            % Outputs: 
            % - (none)
            p = inputParser;

            addParameter(p, 'remove_mean', false, @islogical);
            addParameter(p, 'zscore', false, @islogical);
            addParameter(p, 'time_first', false, @islogical);
            
            parse(p,varargin{:});
            
            remove_mean_opt = p.Results.remove_mean;
            zscore_opt = p.Results.zscore;
            time_first = p.Results.time_first;
            
            if zscore_opt
                remove_mean_opt = true; % Must also remove the mean for z-scoring
            end
            if iscell(Y)
                if time_first
                    YCat = cat(1, Y{:});
                else
                    YCat = cat(2, Y{:});
                end
            else
                YCat = Y;
            end
            if ~time_first
                YCat = YCat.';
            end
            yDim = size(YCat, 2);
            yMean = zeros(1, yDim);
            yStd = ones(1, yDim);
            if remove_mean_opt
                yMean = mean(YCat, 'omitnan');
            end
            if zscore_opt
                yStd = std(YCat, 1, 'omitnan');
                if any(yStd == 0)
                    warning('%d dimension(s) of y (out of %d) are flat. Will skip scaling to unit variance for those dimensions.', sum(yStd==0), numel(yStd))
                end
                if all(yStd == 0)
                    zscore_opt = false;
                end
            end
            obj.remove_mean = remove_mean_opt;
            obj.zscore = zscore_opt;
            obj.dataMean = yMean;
            obj.dataStd = yStd;
        end
        
        function yMean = get_mean(obj, time_first)
            % Returns the mean, but transposes it if needed
            % Inputs: 
            % Any of the following can be provided as a key-value pair.
            % - (1) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) yMean: mean of the data, approtpiately transposed.
            if nargin < 2, time_first = true; end
            if time_first
                yMean = obj.dataMean;
            else
                yMean = obj.dataMean.';
            end
        end
        
        function yStd = get_std(obj, time_first)
            % Returns the std, but transposes it if needed
            % Inputs: 
            % - (1) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) yStd: mean of the data, approtpiately transposed.
            if nargin < 2, time_first = true; end
            if time_first
                yStd = obj.dataStd;
            else
                yStd = obj.dataStd.';
            end
        end
        
        function Y = apply_segment(obj, Y, time_first)
            % Applies the preprocessing on new data segment
            % Inputs: 
            % - (1) Y (numpy array): Input data. First dimension must be  
            %           time and the second dimension is the data. 
            % - (2) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) Y: preprocessed data.
            if nargin < 3, time_first = true; end
            if obj.remove_mean
                Y = Y - obj.get_mean(time_first);
            end
            if obj.zscore
                okDims = obj.dataStd > 0;
                yStd = obj.get_std(time_first);
                if time_first
                    Y(:, okDims) = Y(:, okDims) ./ yStd(:, okDims);
                else
                    Y(okDims, :) = Y(okDims, :) ./ yStd(okDims, :);
                end
            end
        end
        
        function Y = apply(obj, Y, time_first)
            % Applies the preprocessing on new data
            % Inputs: 
            % - (1) Y (numpy array): Input data. First dimension must be  
            %           time and the second dimension is the data. Can be a 
            %           cell array of multiple data segments.
            % - (2) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) Y: preprocessed data.
            if nargin < 3, time_first = true; end
            if iscell(Y)
                Y = arrayfun(@(YThis)( obj.apply_segment(YThis{1}, time_first) ), Y, 'UniformOutput', false);
            else
                Y = obj.apply_segment(Y, time_first);
            end
        end
        
        function Y = apply_inverse_segment(obj, Y, time_first)
            % Applies inverse of the preprocessing on new data segment 
            % (i.e. undoes the preprocessing)
            % Inputs: 
            % - (1) Y (numpy array): Input data. First dimension must be  
            %           time and the second dimension is the data. 
            % - (2) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) Y: data with preprocessing undone.
            if nargin < 3, time_first = true; end
            if obj.zscore
                okDims = obj.dataStd > 0;
                yStd = obj.get_std(time_first);
                if time_first
                    Y(:, okDims) = Y(:, okDims) .* yStd(:, okDims);
                else
                    Y(okDims, :) = Y(okDims, :) .* yStd(okDims, :);
                end
            end
            if obj.remove_mean
                Y = Y + obj.get_mean(time_first);
            end
        end
        
        function Y = apply_inverse(obj, Y, time_first)
            % Applies inverse of the preprocessing on new data (i.e. undoes 
            % the preprocessing)
            % Inputs: 
            % - (1) Y (numpy array): Input data. First dimension must be  
            %           time and the second dimension is the data. Can be a 
            %           cell array of multiple data segments.
            % - (2) time_first (bool, optional): If true, will return the 
            %           mean a row vector, otherwise returns it as a row 
            %           vector. Defaults to True.
            % Outputs: 
            % - (1) Y: data with preprocessing undone.
            if nargin < 3, time_first = true; end
            if iscell(Y)
                Y = arrayfun(@(YThis)( obj.apply_inverse_segment(YThis{1}, time_first) ), Y, 'UniformOutput', false);
            else
                Y = obj.apply_inverse_segment(Y, time_first);
            end
        end
        
    end
    
    methods(Static)
        
    end
    
end
