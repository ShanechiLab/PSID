% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add PSID to the path (or call init.m)
addpath(genpath('../source')); 

%% (1) IPSID examples
% ###################################
% ###################################
%% Load data
data = load('./sample_model_IPSID.mat');
trueSys = data.trueSys;
trueSys.Dz = trueSys.dz;
% This is an example model (shown in Fig. 2A) with 
% (a) 2 behaviorally relevant latent states x_k^(1) (corresponding to intrinsic behaviorally relevant dynamics), 
% (b) 4 other latent states x_k^(2) (corresponding to other intrinsic dynamics),
% (c) 2 states that drive the external input (corresponding to input dynamics)

% Let's first generate some sample data from this model
rng(42) % For exact reproducibility

N = 2e5; % Total number of samples, the more data you have,
         % the more accurate the identification will be
         
% Generating dynamical input u   
[u, ~] = generateLSSMRealization(data.uSys, 1:N);
Au = data.uSys.a;
[y, x] = generateLSSMRealization(trueSys, 1:N, [], [], [], u);
z = (trueSys.Cz * x')' + (trueSys.Dz * u')';
% Add some z dynamics that are not encoded in y (i.e. epsilon)
[~, x_eps] = generateLSSMRealization(data.epsSys, 1:N);
z = z + (data.epsSys.Cz * x_eps')';

allYData = y;
allZData = z;
allUData = u;

% Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean. 
% To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data.
% We will next remove the mean from data and apply it back after the inference.
YMean = 10*randn(1, size(allYData, 2));
ZMean = 10*randn(1, size(allZData, 2));
UMean = 10*randn(1, size(allUData, 2));
allYData = allYData + YMean;
allZData = allZData + ZMean;
allUData = allUData + UMean;
% Also reflect this in the true model:
trueSys.YPrepModel = PrepModel('mean', YMean, 'remove_mean', true);
trueSys.ZPrepModel = PrepModel('mean', ZMean, 'remove_mean', true);
trueSys.UPrepModel = PrepModel('mean', UMean, 'remove_mean', true);


% Separate data into training and test data:
trainInds = (1:round(0.5*size(allYData, 1)))';
testInds = ((1+trainInds(end)):size(allYData, 1))';
yTrain = allYData(trainInds, :);
yTest = allYData(testInds, :);
zTrain = allZData(trainInds, :);
zTest = allZData(testInds, :);
uTrain = allUData(trainInds, :);
uTest = allUData(testInds, :);
%% (Example 1) IPSID can be used to dissociate and extract only the 
% intrinsic behaviorally relevant latent states (with nx = n1 = 2)
idSys1 = IPSID(yTrain', zTrain', uTrain', 2, 2, 10);

% Predict behavior using the learned model
[zTestPred1, ~, xTestPred1] = PSIDPredict(idSys1, yTest, uTest);
% zTestPred1 = zTestPred1 + ZMean;
% Compute decoding accuracy
R2 = evalPrediction(zTest, zTestPred1, 'R2');

% Predict behavior using the true model for comparison
[zTestPredIdeal, yTestPredIdeal, xTestIdeal] = PSIDPredict(trueSys, yTest, uTest);
% zTestPredIdeal = zTestPredIdeal + ZMean;
% yTestPredIdeal = yTestPredIdeal + YMean;
R2Ideal = evalPrediction(zTest, zTestPredIdeal, 'R2');

fprintf('Behavior decoding R2:\n  IPSID => %.3g, Ideal using true model => %.3g\n', mean(R2), mean(R2Ideal));
%% (Example 2) Optionally, IPSID can additionally also learn the 
% behaviorally irrelevant latent states (with nx = 6, n1 = 2)
idSys2 = IPSID(yTrain', zTrain', uTrain', 6, 2, 10);

% In addition to ideal behavior decoding, this model will also have ideal neural self-prediction 
[zTestPred2, yTestPred2, xTestPred2] = PSIDPredict(idSys2, yTest, uTest);

yR2 = evalPrediction(yTest, yTestPred2, 'R2');
yR2Ideal = evalPrediction(yTest, yTestPredIdeal, 'R2');
fprintf('Neural self-prediction R2:\n  IPSID => %.3g, Ideal using true model = %.3g\n', mean(yR2), mean(yR2Ideal));
%% Plot the true and identified eigenvalues

% (Example 1) Eigenvalues when only learning behaviorally relevant states
idEigs1 = eig(idSys1.A);

% (Example 2) Additional eigenvalues when also learning behaviorally irrelevant states
% The identified model is already in form of Eq. 1, with behaviorally irrelevant states 
% coming as the last 4 dimensions of the states in the identified model
idEigs2 = eig(idSys2.A(3:end, 3:end)); 

relevantDims = trueSys.zDims; % Dimensions that drive both behavior and neural activity
irrelevantDims = find(~ismember(1:size(trueSys.a,1), trueSys.zDims)); % Dimensions that only drive the neural activity
trueEigsRelevant = eig(trueSys.a(relevantDims, relevantDims));
trueEigsIrrelevant = eig(trueSys.a(irrelevantDims, irrelevantDims));
trueEigsInput = eig(Au); 

figure; zplane([], []); ax = gca; hold(ax, 'on');
h1 = scatter(ax, real(trueEigsInput), imag(trueEigsInput), 'o', 'MarkerEdgeColor', '#800080', 'DisplayName', 'Input eigenvalues');
h2 = scatter(ax, real(trueEigsIrrelevant), imag(trueEigsIrrelevant), 'o', 'MarkerEdgeColor', '#FF5733', 'DisplayName', 'Other neural eigenvalues');
h3 = scatter(ax, real(trueEigsRelevant), imag(trueEigsRelevant), 'o', 'MarkerEdgeColor', '#50C878', 'DisplayName', 'Behaviorally relevant neural eigenvalues');
h4 = scatter(ax, real(idEigs1), imag(idEigs1), 'x', 'LineWidth', 1, 'MarkerEdgeColor', '#138a33' , 'DisplayName', 'IPSID Identified (stage 1)');
h5 = scatter(ax, real(idEigs2), imag(idEigs2), 'x', 'LineWidth', 1, 'MarkerEdgeColor', '#b04c1a', 'DisplayName', '(optional) IPSID Identified (stage 2)');
title(ax, 'True and identified eigevalues')
legend(ax, [h1, h2, h3, h4, h5], 'Location', 'EO');
%% (2) IPSID example with the additional steps
% ####################################################
% ####################################################
%% Load data
data = load('./sample_model_IPSID_add_step.mat');
trueSys = data.trueSys;
% This is an example model (shown in Fig. 3) with 
% (a) 2 behaviorally relevant latent states, x_k^(1), encoded in neural activity y_k (corresponding to intrinsic behaviorally relevant neural dynamics), 
% (b) 2 other latent states, x_k^(2), encoded in neural activity y_k (corresponding to other intrinsic dynamics),
% (c) 2 states that drive the external input (corresponding to input dynamics)
% (d) 2 behaviorally relevant latent states, x_k^(3), driven by the input u_k but not encoded in neural activity y_k

% Let's first generate some sample data from this model
rng(42) % For exact reproducibility

N = 2e5; % Total number of samples, the more data you have,
         % the more accurate the identification will be

% Generating dynamical input u   
[u, ~] = generateLSSMRealization(data.uSys, 1:N);
Au = data.uSys.a;

% Generate the main realization given input u
[y, x] = generateLSSMRealization(trueSys, 1:N, [], [], [], u);
z = (trueSys.Cz * x')';

allYData = y;
allZData = z;
allUData = u;

% Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean. 
% To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data.
% We will next remove the mean from data and apply it back after the inference.
YMean = 10*randn(1, size(allYData, 2));
ZMean = 10*randn(1, size(allZData, 2));
UMean = 10*randn(1, size(allUData, 2));
allYData = allYData + YMean;
allZData = allZData + ZMean;
allUData = allUData + UMean;
% Also reflect this in the true model:
trueSys.YPrepModel = PrepModel('mean', YMean, 'remove_mean', true);
trueSys.ZPrepModel = PrepModel('mean', ZMean, 'remove_mean', true);
trueSys.UPrepModel = PrepModel('mean', UMean, 'remove_mean', true);

% Separate data into training and test data:
trainInds = (1:round(0.5*size(allYData, 1)))';
testInds = ((1+trainInds(end)):size(allYData, 1))';
yTrain = allYData(trainInds, :);
yTest = allYData(testInds, :);
zTrain = allZData(trainInds, :);
zTest = allZData(testInds, :);
uTrain = allUData(trainInds, :);
uTest = allUData(testInds, :);
%% (Example 3) IPSID with additional steps can be used to further dissociate the intrinsic behaviorally relevant neural dynamics that encoded in neural activity from those that are not 

% all latent states [x1;x2,x3] (with nx = 6, n1 = 2, n3 = 2)
idSys3 = IPSID(yTrain', zTrain', uTrain', 6, 2, 10, 'remove_nonYrelated_fromX1', true, 'n_pre', 4, 'n3', 2); % n_pre should be equal to true n1+true n2

[zTestPred3, yTestPred3, ~] = PSIDPredict(idSys3, yTest, uTest);
% Compute decoding and neural self-prediction accuracy
R2 = evalPrediction(zTest, zTestPred3, 'R2');
yR2 = evalPrediction(yTest, yTestPred3, 'R2');

% For comparison, let's also learn a model without the additional step 2 (only [x1;x2])
idSys3_low_dim = IPSID(yTrain', zTrain', uTrain', 4, 2, 10, 'remove_nonYrelated_fromX1', true, 'n_pre', 4, 'n3', 0); % n_pre should be equal to true n1+true n2
[zTestPred3_low_dim, yTestPred3_low_dim, ~] = PSIDPredict(idSys3_low_dim, yTest, uTest);
% Compute decoding and neural self-prediction accuracy
R2_low_dim = evalPrediction(zTest, zTestPred3_low_dim, 'R2');
yR2_low_dim = evalPrediction(yTest, yTestPred3_low_dim, 'R2');

% Predict using the true model for comparison
[zTestPredIdeal, yTestPredIdeal, ~] = PSIDPredict(trueSys, yTest, uTest);
yR2Ideal = evalPrediction(yTest, yTestPredIdeal, 'R2');
R2Ideal = evalPrediction(zTest, zTestPredIdeal, 'R2');

fprintf('Decoding R2:\n  IPSID => %.3g, IPSID (without additional step 2) => %.3g, Ideal using true model = %.3g\n', mean(R2), mean(R2_low_dim), mean(R2Ideal));
fprintf('Neural self-prediction R2:\n  IPSID => %.3g, IPSID (without additional step 2) => %.3g, Ideal using true model = %.3g\n', mean(yR2), mean(yR2_low_dim), mean(yR2Ideal));
%% Plot the true and identified eigenvalues for IPSID with the additional steps

% Intrinsic behaviorally relevant eigenvalues encoded in neural activity
idEigs1 = eig(idSys3_low_dim.A(1:2,1:2));
% Other intrinsic eigenvalues encoded in neural activity
idEigs2 = eig(idSys3_low_dim.A(3:4, 3:4)); 
% Behaviorally relevant eigenvalues not encoded in neural activity
idEigs3 = eig(idSys3.A(5:6, 5:6)); 

relevantDims = trueSys.zDims; % Dimensions that drive both behavior and neural activity
irrelevantDims = 3:4; % Dimensions that only drive the neural activity
trueEigsRelevant = eig(trueSys.a(relevantDims, relevantDims));
trueEigsIrrelevant = eig(trueSys.a(irrelevantDims, irrelevantDims));
trueEigsInput = eig(Au); 
trueEigsNonEncoded = eig(trueSys.a(5:6, 5:6));

figure; zplane([], []); ax = gca; hold(ax, 'on');
h1 = scatter(ax, real(trueEigsInput), imag(trueEigsInput), 'o', 'MarkerEdgeColor', '#800080', 'DisplayName', 'Input eigenvalues');
h2 = scatter(ax, real(trueEigsIrrelevant), imag(trueEigsIrrelevant), 'o', 'MarkerEdgeColor', '#FF5733', 'DisplayName', 'Other neural eigenvalues');
h3 = scatter(ax, real(trueEigsRelevant), imag(trueEigsRelevant), 'o', 'MarkerEdgeColor', '#50C878', 'DisplayName', 'Behaviorally relevant neural eigenvalues');
h4 = scatter(ax, real(trueEigsNonEncoded), imag(trueEigsNonEncoded), 'o', 'MarkerEdgeColor', '#000000', 'DisplayName', 'Behaviorally relevant not encoded in neural activity eigenvalues');
h5 = scatter(ax, real(idEigs1), imag(idEigs1), 'x', 'MarkerEdgeColor', '#138a33', 'DisplayName', 'IPSID Identified (stage 1)');
h6 = scatter(ax, real(idEigs2), imag(idEigs2), 'x', 'MarkerEdgeColor', '#b04c1a', 'DisplayName', '(optional) IPSID Identified (stage 2)');
h7 = scatter(ax, real(idEigs3), imag(idEigs3), 'x', 'MarkerEdgeColor', '#000000', 'DisplayName', '(optional) IPSID Identified in optional additional step 2');

title(ax, 'True and identified eigevalues')
legend(ax, [h1, h2, h3, h4, h5, h6, h7], 'Location', 'EO');