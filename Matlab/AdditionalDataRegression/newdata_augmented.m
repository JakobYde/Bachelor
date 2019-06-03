%% Load data from CSV files
EUL_train = csvread('TrainingDataEUL.csv');
CRP_train = csvread('TrainingDataCRP.csv');
Y_train = csvread('TrainingDataY.csv');

test_data = csvread('NewDataRegression.csv');

patient_number = test_data(:,1);
CRP_test = test_data(:,2);
Y_test = test_data(:,3);
EUL_test = test_data(:,4:length(test_data(1,:)));
%%
sortedCRP_train = sort(CRP_train);
sortedEUL_train = sort(EUL_train);
%% Create X for training
[m, n] = size(EUL_train);

for patient = 1:m
    for joint = 1:n
        if EUL_train(patient, joint) == -1
            sortedEUL_without_unknowns = sortedEUL_train(sortedEUL_train(:,joint)~=-1, joint);
            EUL_train(patient, joint) = sortedEUL_without_unknowns(round(mean(find(sortedCRP_train == CRP_train(patient)))/length(CRP_train) * length(sortedEUL_without_unknowns)));
        end
    end
end

% Concatenate EUL and CRP
X_train = [EUL_train CRP_train];

%% Create X for testing
[m, n] = size(EUL_test);

for patient = 1:m
    sortedCRP = sort([sortedCRP_train; CRP_test(patient)]);
    sortedEUL = sort([sortedEUL_train; EUL_test(patient,:)]);
    for joint = 1:n
        if EUL_test(patient, joint) == -1
            sortedEUL_without_unknowns = sortedEUL(sortedEUL(:,joint)~=-1, joint);
            EUL_test(patient, joint) = sortedEUL_without_unknowns(round(mean(find(sortedCRP == CRP_test(patient)))/length(sortedCRP) * length(sortedEUL_without_unknowns)));
        end
    end
end

% Concatenate EUL and CRP
X_test = [EUL_test CRP_test];

%% Regression

%Calculate the beta-vector using linear multivariable regression
[beta]  = mvregress(X_train, Y_train);

%Use both hands from each patient to estimate the DAS28 score
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

%% Regression

%Calculate the beta-vector with no negative values,
%using linear multivariable regression
[beta]  = lsqnonneg(X_train, Y_train);

%Use both hands from each patient to estimate the DAS28 score
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