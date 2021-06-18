% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testPrepModel Tests the PrepModel class

%% Main function to generate tests
function tests = testPrepModel
    tests = functiontests(localfunctions);
end

%% Test Functions
function testThatPrepModelWorksForSingleSegmentData(testCase)
    arg_sets = {
        { 'remove_mean', true  , 'zscore', true  }, ...
        { 'remove_mean', false , 'zscore', true  }, ...
        { 'remove_mean', true  , 'zscore', false }, ...
        { 'remove_mean', false , 'zscore', false } ...
    };
    
    for ai = 1:numel(arg_sets)
        args = arg_sets{ai};
        numTests = 100;
        for ti = 1:numTests
            n_dim = randi(10);
            
            trueMean = 10 * randn(1, n_dim);
            trueStd = 10 * randn(1, n_dim);
            
            for tfi = 1:2
                if tfi == 1
                    time_first = true;
                else
                    time_first = false;
                end
                remove_mean = args{2};
                zscore = args{4};
                
                n_samples = 10 + randi(100);
                data = randn(n_samples, n_dim) .* trueStd + trueMean;
                if ~time_first
                    data = data.';
                end
                sm = PrepModel();
                sm.fit(data, 'remove_mean', remove_mean, 'zscore', zscore, 'time_first', time_first);
                newData = sm.apply(data, time_first);
                if time_first
                    newDataMean = mean(newData, 1);
                    newDataStd = std(newData, 1, 1);
                else
                    newDataMean = mean(newData, 2);
                    newDataStd = std(newData, 1, 2);
                end
                
                if remove_mean || zscore
                    verifyLessThan(testCase, norm( newDataMean - zeros(size(newDataMean))), 1e5*eps );
                end
                
                if zscore
                    verifyLessThan(testCase, norm( newDataStd - ones(size(newDataMean))), 1e5*eps );
                end
                
                recoveredData = sm.apply_inverse(newData, time_first);
                
                verifyLessThan(testCase, norm( recoveredData - data), 1e4*eps );
            end
            
        end
    end
end

function testThatPrepModelWorksForMultiSegmentData(testCase)
    arg_sets = {
        { 'remove_mean', true  , 'zscore', true  }, ...
        { 'remove_mean', false , 'zscore', true  }, ...
        { 'remove_mean', true  , 'zscore', false }, ...
        { 'remove_mean', false , 'zscore', false } ...
    };
    
    for ai = 1:numel(arg_sets)
        args = arg_sets{ai};
        numTests = 100;
        for ti = 1:numTests
            n_dim = randi(10);
            
            trueMean = 10 * randn(1, n_dim);
            trueStd = 10 * randn(1, n_dim);
            
            n_segments = randi(10);
            
            for tfi = 1:2
                if tfi == 1
                    time_first = true;
                else
                    time_first = false;
                end
                remove_mean = args{2};
                zscore = args{4};
                
                data = {};
                for si = 1:n_segments
                    n_samples = 10 + randi(100);
                    dataThis = randn(n_samples, n_dim) .* trueStd + trueMean;
                    if ~time_first
                        dataThis = dataThis.';
                    end
                    data = cat(1, data, dataThis);
                end
                sm = PrepModel();
                sm.fit(data, 'remove_mean', remove_mean, 'zscore', zscore, 'time_first', time_first);
                newData = sm.apply(data, time_first);
                if time_first
                    newDataCat = cat(1, newData{:});
                    newDataMean = mean(newDataCat, 1);
                    newDataStd = std(newDataCat, 1, 1);
                else
                    newDataCat = cat(2, newData{:});
                    newDataMean = mean(newDataCat, 2);
                    newDataStd = std(newDataCat, 1, 2);
                end
                
                if remove_mean || zscore
                    verifyLessThan(testCase, norm( newDataMean - zeros(size(newDataMean))), 1e5*eps );
                end
                
                if zscore
                    verifyLessThan(testCase, norm( newDataStd - ones(size(newDataMean))), 1e5*eps );
                end
                
                recoveredData = sm.apply_inverse(newData, time_first);
                
                for si = 1:n_segments 
                    verifyLessThan(testCase, norm( recoveredData{si} - data{si}), 1e4*eps );
                end
            end
            
        end
    end
end

%% Optional file fixtures
function setupOnce(testCase)  % do not change function name
% set a new path, for example
addpath(genpath('../'));
end

function teardownOnce(testCase)  % do not change function name
% change back to original path, for example
end

%% Optional fresh fixtures  
function setup(testCase)  % do not change function name
% open a figure, for example
end

function teardown(testCase)  % do not change function name
% close figure, for example
end