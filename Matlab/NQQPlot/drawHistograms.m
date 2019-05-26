%filtered = readtable('filtered.csv')

n = table2array(filtered(:,12));
nn = table2array(filtered(:,2));

temp = nn;
temp = [temp; temp(length(temp)) + 1];

arr = [];

figure(1);
subplot(4,2,1);
qqplot(n);
title('All models')  ;

for i = 1:7
    index = find(temp - i, 1, 'first') - 1;
    data = n(1:index);
    data(length(data))
    arr = [arr, {n(1:index)}];
    temp = temp(index + 1:length(temp));
    
    subplot(4,2,i + 1);
    qqplot(data);
    title(join([num2str(i),'-layer'])) ; 
end

figure(2)
scatter(nn, n)

