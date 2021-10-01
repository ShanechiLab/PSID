- [PSID: Preferential subspace identification <br/> [MATLAB implementation]](#psid-preferential-subspace-identification--matlab-implementation)
- [Publication](#publication)
- [Usage guide](#usage-guide)
  - [Initialization](#initialization)
  - [Main learning function](#main-learning-function)
  - [Extracting latent states using learned model](#extracting-latent-states-using-learned-model)
  - [Required preprocessing](#required-preprocessing)
  - [Choosing the hyperparameters](#choosing-the-hyperparameters)
    - [How to pick the state dimensions nx and n1?](#how-to-pick-the-state-dimensions-nx-and-n1)
    - [How to pick the horizon `i`?](#how-to-pick-the-horizon-i)
- [Example script](#example-script)
- [Change Log](#change-log)
- [Licence](#licence)

# PSID: Preferential subspace identification <br/> [MATLAB implementation]

For Python implementation see http://github.com/ShanechiLab/PyPSID

Given signals y_t (e.g. neural signals) and z_t (e.g behavior), PSID learns a dynamic model for y_t while prioritizing the dynamics that are relevant to z_t. 

# Publication
For the derivation of PSID and results in real neural data see the paper below.

Omid G. Sani, Hamidreza Abbaspourazad, Yan T. Wong, Bijan Pesaran, Maryam M. Shanechi. *Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification*. Nature Neuroscience, 24, 140–149 (2021). https://doi.org/10.1038/s41593-020-00733-0

View-only full-text link: https://rdcu.be/b993t

Original preprint: https://doi.org/10.1101/808154

You can also find a summary of the paper in the following Twitter thread:
https://twitter.com/MaryamShanechi/status/1325835609345122304


# Usage guide
## Initialization
Add the source directory and its subdirectories to the path. You can run init.m to do this.

## Main learning function
The main function for the MATLAB implementation is [source/PSID.m](source/PSID.m). A complete usage guide is available in the function. The following shows an example case:
```
idSys = PSID(y, z, nx, n1, i);
```
Inputs:
- y and z are time x dimension matrices with neural (e.g. LFP signal powers or spike counts) and behavioral data (e.g. joint angles, hand position, etc), respectively. 
- nx is the total number of latent states to be identified.
- n1 is the number of states that are going to be dedicated to behaviorally relevant dynamics.
- i is the subspace horizon used for modeling. 

Output:
- idSys: a structure containing all model parameters (A, Cy, Cz, etc). For a full list see the code.

## Extracting latent states using learned model
Once a model is learned using PSID, you can apply the model to new data (i.e. run the associated Kalman filter) as follows:
```
[zPred, yPred, xPred] = PSIDPredict(idSys, y);
```
Input:
- y: neural activity time series (time x dimension)

Outputs:
- zPred: one-step ahead prediction of behavior (if any)
- yPred: one-step ahead prediction of neural activity
- xPred: Extracted latent state

## Required preprocessing
A required preprocessing when using PSID is to remove the mean of neural/behavior signals and if needed, add them back to predictions after learning the model. Starting from version 1.1.0, Python and MATLAB PSID libraries automatically do this by default so that users won't need to worry about it. Please update to the latest version if you are using an older version.

## Choosing the hyperparameters
### How to pick the state dimensions nx and n1?
nx determines the total dimension of the latent state and n1 determines how many of those dimensions will be prioritizing the inclusion of behaviorally relevant neural dynamics (i.e. will be extracted using stage 1 of PSID). So the values that you would select for these hyperparameters depend on the goal of modeling and on the data. Some examples use cases are:

If you want to perform dimension reduction, nx will be your desired target dimension. For example, to reduce dimension to 2 to plot low-dimensional visualizations of neural activity, you would use nx=2. Now if you want to reduce dimension while preserving as much behaviorally relevant neural dynamics as possible, you would use n1=nx.
If you want to find the best fit to data overall, you can perform a grid search over values of nx and n1 and pick the value that achieves the best performance metric in the training data. For example, you could pick the nx and n1 pair that achieves the best cross-validated behavior decoding in an inner-cross-validation within the training data.

### How to pick the horizon `i`?
The horizon `i` does not affect the model structure and only affects the intermediate linear algebra operations that PSID performs during the learning of the model. Nevertheless, different values of `i` may have different model learning performance. `i` needs to be at least 2, but also also determines the maximum n1 and nx that can be used per:

```
n1 <= nz * i
nx <= ny * i
```

So if you have a low dimensional y_k or z_k (small ny or nz), you typically would choose larger values for `i`, and vice versa. It is also possible to select the best performing `i` via an inner cross-validation approach similar to nx and n1 above. Overall, since `i` affects the learning performance, it is important for reproducibility that the `i` that was used is reported.

For more information, see the notebook(s) referenced in the next section.

# Usage examples
Example simulated data and the script for running PSID on the data is provided in 
[example/example.m](example/example.m)
This script performs PSID model identification and visualizes the learned eigenvalues similar to in Supplementary Fig 1.

The following notebook also contains some examples with the Python implementation:
https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_tutorial.ipynb

# Change Log
You can see the change log in in [ChangeLog.md](https://github.com/ShanechiLab/PSID/blob/main/ChangeLog.md)  

# Licence
Copyright (c) 2020 University of Southern California  
See full notice in [LICENSE.md](LICENSE.md)  
Omid G. Sani and Maryam M. Shanechi  
Shanechi Lab, University of Southern California
