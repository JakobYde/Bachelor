%% Starting data
DAS28_train = csvread('TrainingDataY.csv');
DAS28_test = csvread('TestingDataY.csv');

DAS28_mean = mean(DAS28_train)

Error_random_MAE = mean(abs(DAS28_mean - DAS28_test))

Error_random_MSE = mean((DAS28_mean - DAS28_test).^2)

%% Additional data

DAS28_train = csvread('TrainingDataY.csv');
additional_data = csvread('NewDataRegression.csv');

DAS28_test = additional_data((1:2:length(additional_data(:,3))),3);

DAS28_mean = mean(DAS28_train)

Error_random_MAE = mean(abs(DAS28_mean - DAS28_test))

Error_random_MSE = mean((DAS28_mean - DAS28_test).^2)