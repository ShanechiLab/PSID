# PSID: Preferential subspace identification <br/> [MATLAB implementation]

For Python implementation see http://github.com/ShanechiLab/PyPSID

Given signals y_t (e.g. neural signals) and z_t (e.g behavior), PSID learns a dynamic model for y_t while prioritizing the dynamics that are relevant to z_t. 

For the derivation and results in real neural data see the paper below.

## Publication: 
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

# Example script
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
