% test code for learning the dynamics of data
% Given x[n], a matrix of data (considered as state vectors) in NxT shape,
%   we seek x[n+1] = A[n].x[n] + w[n]
%   A[n] is found by a least squares error fit over a look back window of
%   length wlen
% 
% Reza Sameni
% June 2022
% Dependencies: Uses data and functions from OSET: https://github.com/alphanumericslab/OSET.git

clc
clear;
close all;

load('FOETAL_ECG.dat'); data = FOETAL_ECG(:,2:end)'; clear FOETAL_ECG; fs = 250;
% load patient165_s0323lre; data = data(:, 2:6)'; fs = 1000;

N = size(data, 1); % Num Channels
T = size(data, 2); % Num Samples
wlen = round(0.1 * fs); % Lookback window length

A = zeros(N, N, T); % A[n]
lambda = zeros(N, T); % eigenvalues of A[n]
Q = zeros(N, N, T); % covariance of w[n]
q = zeros(N, T); % diagonal entries of Q
for n = wlen + 1 : T
    X_current = data(:, n - wlen + 1 : n);
    X_previous = data(:, n - wlen : n - 1);

    A(:, :, n) =  X_current / X_previous;
    w = X_current - A(:, :, n) * X_previous;
    Q(:, :, n) = cov(w');
    q(:, n) = diag(Q(:, :, n));
    [~, lambda(:, n)] = eig(A(:, :, n), 'vector');
end

PlotECG(data, N, 'b', fs, 'Input data x[n]');

PlotECG(sort(abs(lambda), 'descend'), N, 'b', fs, 'Eigenvalues of A[n]');

PlotECG(sort(q, 'descend'), N, 'b', fs, 'variance of the residues w[n]');
