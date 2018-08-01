clear;
clc;

% initialization

addpath('src');

datapath = 'dataset/6ng/';
alpha = 1;
beta = 1;
lambda = 1e-4;
batchsize = 200;
stepsize = 0.1;
momentum = 0.9;
maxepoch = 200;
decay = 1;
acttype = 'sigmoid';
gpuidx = 1;
tr_ratio = 0.9;
vis = 1;
dmnetype = 'pd';

% run dmne algorithm
rundmne(datapath, alpha, beta, lambda, batchsize, stepsize, momentum, ...
    maxepoch, decay, acttype, gpuidx, tr_ratio, vis, dmnetype);
