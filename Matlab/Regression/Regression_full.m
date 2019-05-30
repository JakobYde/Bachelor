%% Load data from CSV files
EUL_train = csvread('TrainingDataEUL.csv');
CRP_train = csvread('TrainingDataCRP.csv');
DAS28_train = csvread('TrainingDataY.csv');

EUL_test = csvread('TestingDataEUL.csv');
CRP_test = csvread('TestingDataCRP.csv');
DAS28_test = csvread('TestingDataY.csv');

%% Concatenate euler-omeract and CRP scores and remove those that aren't full for the training set.
[m, n] = size(EUL_train);
X_train = [];
Y_train = [];
for i = 1:m
    FullSet = true;
    for j = 1:n
        if EUL_train(i, j) == -1
            FullSet = false;
            break
        end
    end
    if FullSet == true
        X_new = [EUL_train(i, :) CRP_train(i)];
        X_train = [X_train; X_new];
        Y_train = [Y_train; DAS28_train(i)];
    end
end

%% Concatenate euler-omeract and CRP scores and remove those that aren't full for the test set.
[m, n] = size(EUL_test);
X_test = [];
Y_test = [];
for i = 1:m
    FullSet = true;
    for j = 1:n
        if EUL_test(i, j) == -1
            FullSet = false;
            break
        end
    end
    if FullSet == true
        X_new = [EUL_test(i, :) CRP_test(i)];
        X_test = [X_test; X_new];
        Y_test = [Y_test; DAS28_test(i)];
    end
end

%% Regression using only full datasets

%Calculate the beta-vector using linear multivariable regression
[beta]  = mvregress(X_train, Y_train);

%Plot the estimations against the real values from the test set.
x = 1:length(Y_test);
estimates = X_test*beta;
f1 = figure('Name', 'mvregress');
hold on
scatter(x, Y_test, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(Y_test)+1]) 
ylim([0 6])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(Y_test))
for i = x
    line([i i], [Y_test(i) estimates(i)]);
end
hold off

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(Y_test - X_test*beta))
mse = mean((Y_test - X_test*beta).^2)

%% Regression using only full datasets and positive weights.

%Calculate the beta-vector with no negative values,
%using linear multivariable regression
[beta] = lsqnonneg(X_train, Y_train);

%Plot the estimations against the real values from the test set.
x = 1:length(Y_test);
estimates = X_test*beta;
f2 = figure('Name', 'lsqnonneg');
hold on
scatter(x, Y_test, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(Y_test)+1]) 
ylim([0 6])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(Y_test))
for i = x
    line([i i], [Y_test(i) estimates(i)]);
end
hold off

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(Y_test - X_test*beta))
mse = mean((Y_test - X_test*beta).^2)