% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add PSID to the path (or call init.m)
addpath(genpath('../source')); 

%% Load data
data = load('./sample_model.mat');
trueSys = data.trueSys;
% This is an example model (shown in Supplementary Fig. 1) with 
% (a) 2 behaviorally relevant latent states, 
% (b) 2 behaviorally irrelevant latent states, and 
% (c) 2 states that drive behavior but are not represented in neural activity

% Let's first generate some sample data from this model
rng(42) % For exact reproducibility

N = 2e4; % Total number of samples, the more data you have,
         % the more accurate the identification will be

[y, x] = generateLSSMRealization(trueSys, 1:N);
z = (trueSys.Cz * x')';
% Add some z dynamics that are not encoded in y (i.e. epsilon)
eps = generateLSSMRealization(data.epsSys, 1:N);
z = z + eps;

allYData = y;
allZData = z;

% Given the stable state-space model used by PSID, it is important for the 
% neural/behavior data to be zero-mean. Starting version v1.1.0, PSID by 
% default internally removes the mean from the neural/behavior data and adds
% it back to predictions, so the user does not need to handle this 
% preprocessing. If the data is already zero-mean, this mean-removal will 
% simply subtract and add zeros to signals so everything will still work.
% To cover this general case with data that is not zero-mean, only for this 
% simultion, let's artificially add some non-zero mean to the sample data:
YMean = 10*randn(1, size(allYData, 2));
ZMean = 10*randn(1, size(allZData, 2));
allYData = allYData + YMean;
allZData = allZData + ZMean;
% Also reflect this in the true model:
trueSys.YPrepModel = PrepModel('mean', YMean, 'remove_mean', true);
trueSys.ZPrepModel = PrepModel('mean', ZMean, 'remove_mean', true);


% Separate data into training and test data:
trainInds = (1:round(0.5*size(allYData, 1)))';
testInds = ((1+trainInds(end)):size(allYData, 1))';
yTrain = allYData(trainInds, :);
yTest = allYData(testInds, :);
zTrain = allZData(trainInds, :);
zTest = allZData(testInds, :);
%% (Example 1) PSID can be used to dissociate and extract only the 
% behaviorally relevant latent states (with nx = n1 = 2)
idSys1 = PSID(yTrain, zTrain, 2, 2, 10);
% You can also use the time_first=False argument if time is the second dimension:
% idSys1 = PSID(yTrain', zTrain', 2, 2, 10, [], [], false);

% Predict behavior using the learned model
[zTestPred1, ~, xTestPred1] = PSIDPredict(idSys1, yTest);

% Compute decoding accuracy
R2 = evalPrediction(zTest, zTestPred1, 'R2');

% Predict behavior using the true model for comparison
[zTestPredIdeal, yTestPredIdeal, xTestIdeal] = PSIDPredict(trueSys, yTest);
R2Ideal = evalPrediction(zTest, zTestPredIdeal, 'R2');

fprintf('Behavior decoding R2:\n  PSID => %.3g, Ideal using true model => %.3g\n', mean(R2), mean(R2Ideal));
%% (Example 2) Optionally, PSID can additionally also learn the 
% behaviorally irrelevant latent states (with nx = 4, n1 = 2)
idSys2 = PSID(yTrain, zTrain, 4, 2, 10);

% In addition to ideal behavior decoding, this model will also have ideal neural self-prediction 
[zTestPred2, yTestPred2, xTestPred2] = PSIDPredict(idSys2, yTest);

yR22 = evalPrediction(yTest, yTestPred2, 'R2');
yR2Ideal = evalPrediction(yTest, yTestPredIdeal, 'R2');
fprintf('Neural self-prediction R2:\n  PSID => %.3g, Ideal using true model = %.3g\n', mean(yR22), mean(yR2Ideal));

%% (Example 3) PSID can be used if data is available in discontinuous segments (e.g. different trials)
% In this case, y and z data segments must be provided as elements of a cell array
% Trials do not need to have the same number of samples
% Here, for example assume that trials start at every 1000 samples.
% And each each trial has a random length of 500 to 900 samples
trialStartInds = (1:1000:(size(allYData, 1)-1000))';
trialDurRange = [900 990];
trialDur = trialDurRange(1)-1 + randi(diff(trialDurRange)+1, size(trialStartInds));
trialInds = arrayfun( @(ti)( (trialStartInds(ti)-1+(1:trialDur(ti)))' ), (1:numel(trialStartInds))', 'UniformOutput', false );
yTrials = arrayfun( @(tInds)( allYData(tInds{1}, :) ), trialInds, 'UniformOutput', false );
zTrials = arrayfun( @(tInds)( allZData(tInds{1}, :) ), trialInds, 'UniformOutput', false );

% Separate data into training and test data:
trainInds = (1:round(0.5*numel(yTrials)))';
testInds = ((1+trainInds(end)):numel(yTrials))';
yTrain = yTrials(trainInds, :);
yTest = yTrials(testInds, :);
zTrain = zTrials(trainInds, :);
zTest = zTrials(testInds, :);

idSys3 = PSID(yTrain, zTrain, 2, 2, 10);

% yTestCat = cell2mat( yTest ); % Data can also be concatenated for
                % decoding if taking last state in a previous trial as the 
                % initial state in the next trial makes sense
[zTestPred1, ~, xTestPred1Cell] = PSIDPredict(idSys3, yTest);
[zTestPredIdealCell, ~, xTestPredIdealCell] = PSIDPredict(trueSys, yTest);

zTestPred1Cat = cell2mat( zTestPred1 );
zTestPredIdealCat = cell2mat( zTestPredIdealCell );
% zTestPred1Cat = zTestPred1;

zTestCat = cell2mat( zTest );
R2TrialBased = evalPrediction(zTestCat, zTestPred1Cat, 'R2');
R2TrialBasedIdeal = evalPrediction(zTestCat, zTestPredIdealCat, 'R2');

fprintf('Behavior decoding R2 (trial-based learning/decoding):\n  PSID => %.3g, Ideal using true model => %.3g\n', mean(R2TrialBased), mean(R2TrialBasedIdeal));

%%
% Plot the true and identified eigenvalues

% (Example 1) Eigenvalues when only learning behaviorally relevant states
idEigs1 = eig(idSys1.A);

% (Example 2) Additional eigenvalues when also learning behaviorally irrelevant states
% The identified model is already in form of Eq. 4, with behaviorally irrelevant states 
% coming as the last 2 dimensions of the states in the identified model
idEigs2 = eig(idSys2.A(3:4, 3:4)); 

relevantDims = trueSys.zDims; % Dimensions that drive both behavior and neural activity
irrelevantDims = find(~ismember(1:size(trueSys.a,1), trueSys.zDims)); % Dimensions that only drive the neural activity
trueEigsRelevant = eig(trueSys.a(relevantDims, relevantDims));
trueEigsIrrelevant = eig(trueSys.a(irrelevantDims, irrelevantDims));
nonEncodedEigs = eig(data.epsSys.a);  % Eigenvalues for states that only drive behavior

figure; zplane([], []); ax = gca; hold(ax, 'on');
h1 = scatter(ax, real(nonEncodedEigs), imag(nonEncodedEigs), 'o', 'MarkerEdgeColor', 'b', 'DisplayName', 'Not encoded in neural signals');
h2 = scatter(ax, real(trueEigsIrrelevant), imag(trueEigsIrrelevant), 'o', 'MarkerEdgeColor', 'r', 'DisplayName', 'Behaviorally irrelevant');
h3 = scatter(ax, real(trueEigsRelevant), imag(trueEigsRelevant), 'o', 'MarkerEdgeColor', 'g', 'DisplayName', 'Behaviorally relevant');
h4 = scatter(ax, real(idEigs1), imag(idEigs1), 'x', 'MarkerEdgeColor', [0 0.5 0], 'DisplayName', 'PSID Identified (stage 1)');
h5 = scatter(ax, real(idEigs2), imag(idEigs2), 'x', 'MarkerEdgeColor', [0.5 0 0], 'DisplayName', '(optional) PSID Identified (stage 2)');
title(ax, 'True and identified eigevalues')
legend(ax, [h1, h2, h3, h4, h5], 'Location', 'EO');
