%% Load data from CSV files
EUL_train = csvread('TrainingDataEUL.csv');
CRP_train = csvread('TrainingDataCRP.csv');
DAS28_train = csvread('TrainingDataY.csv');

test_data = csvread('NewDataRegression.csv');

patient_number = [];
CRP_test = test_data(:,2);
DAS28_test = test_data(:,3);
EUL_test = test_data(:,4:length(test_data(1,:)));

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
        patient_number = [patient_number; test_data(i,1)];
    end
end
%% Regression using only full datasets

%Calculate the beta-vector using linear multivariable regression
[beta]  = mvregress(X_train, Y_train);

%Use both hands from each patient_number to estimate the DAS28 score
estimates = [];
ground_truth = [];
j = 1;
estimates = [estimates;  X_test(1,:)*beta];
ground_truth = [ground_truth; Y_test(1)];
for i = 2:length(Y_test)
    if patient_number(i) == patient_number(i - 1)
        estimates(j) = (estimates(j) + X_test(i,:)*beta)/2;
    elseif patient_number(i) ~= patient_number(i - 1)
        estimates = [estimates; X_test(i,:)*beta];
        ground_truth = [ground_truth; Y_test(i)];
        j = j + 1;
    end
end

%Plot the estimations against the real values from the test set.
x = 1:length(ground_truth);
f1 = figure('Name', 'mvregress');
hold on
scatter(x, ground_truth, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(ground_truth)+1]) 
ylim([min(estimates) 7])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(ground_truth))
for i = x
    line([i i], [ground_truth(i) estimates(i)]);
end
set(gca,'fontsize',14)
hold off

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(Y_test - X_test*beta))
mse = mean((Y_test - X_test*beta).^2)

%% Regression using only full datasets and positive weights.

%Calculate the beta-vector with no negative values,
%using linear multivariable regression
[beta] = lsqnonneg(X_train, Y_train);

%Use both hands from each patient_number to estimate the DAS28 score
estimates = [];
ground_truth = [];
j = 1;
estimates = [estimates;  X_test(1,:)*beta];
ground_truth = [ground_truth; Y_test(1)];
for i = 2:length(Y_test)
    if patient_number(i) == patient_number(i - 1)
        estimates(j) = (estimates(j) + X_test(i,:)*beta)/2;
    elseif patient_number(i) ~= patient_number(i - 1)
        estimates = [estimates; X_test(i,:)*beta];
        ground_truth = [ground_truth; Y_test(i)];
        j = j + 1;
    end
end

%Plot the estimations against the real values from the test set.
x = 1:length(ground_truth);
f1 = figure('Name', 'lsqnonneg');
hold on
scatter(x, ground_truth, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(ground_truth)+1]) 
ylim([min(estimates) 7])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(ground_truth))
for i = x
    line([i i], [ground_truth(i) estimates(i)]);
end
set(gca,'fontsize',14)
hold off

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(Y_test - X_test*beta))
mse = mean((Y_test - X_test*beta).^2)