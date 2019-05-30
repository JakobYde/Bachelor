%% Load data from CSV files
EUL_train = csvread('TrainingDataEUL.csv');
CRP_train = csvread('TrainingDataCRP.csv');
DAS28_train = csvread('TrainingDataY.csv');

EUL_test = csvread('TestingDataEUL.csv');
CRP_test = csvread('TestingDataCRP.csv');
DAS28_test = csvread('TestingDataY.csv');
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
[beta]  = mvregress(X_train, DAS28_train);

%Plot the estimations against the real values from the test set.
x = 1:length(DAS28_test);
estimates = X_test*beta;
f1 = figure('Name', 'mvregress');
hold on
scatter(x, DAS28_test, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(DAS28_test)+1])
ylim([0 6])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(DAS28_test))

for i = 1:length(DAS28_test)
    line([i i], [DAS28_test(i) estimates(i)]);
end

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(DAS28_test - X_test*beta))
mse = mean((DAS28_test - X_test*beta).^2)
hold off

%% Regression

%Calculate the beta-vector with no negative values,
%using linear multivariable regression
[beta]  = lsqnonneg(X_train, DAS28_train);

%Plot the estimations against the real values from the test set.
x = 1:length(DAS28_test);
estimates = X_test*beta;
f2 = figure('Name', 'lsqnonneg');
hold on
scatter(x, DAS28_test, 'green')
scatter(x, estimates, 'red')

%Set plot specifications
legend('Ground truth', 'Estimates','AutoUpdate','off')
xlim([0 length(DAS28_test)+1]) 
ylim([0 6])
xlabel('Sample number')
ylabel('DAS28 score')
set(gca, 'XTick', 1:length(DAS28_test))
for i = 1:length(DAS28_test)
    line([i i], [DAS28_test(i) estimates(i)]);
end
hold off

%Print beta-vector, MAE and MSE
beta
mae = mean(abs(DAS28_test - X_test*beta))
mse = mean((DAS28_test - X_test*beta).^2)