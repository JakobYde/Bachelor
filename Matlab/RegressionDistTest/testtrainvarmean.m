a1 = readtable('CRP_weighted_y_train.csv')
a2 = readtable('CRP_weighted_y_test.csv')
das_wgt = [a1; a2]

a1 = readtable('CRP_weighted_x_train.csv')
a2 = readtable('CRP_weighted_x_test.csv')
crp_wgt = [a1(:,11); a2(:,11)]

a1 = readtable('full_x_train.csv')
a2 = readtable('full_x_test.csv')
crp_raw = [a1(:,11); a2(:,11)]
das_raw = [a1(:,12); a2(:,12)]


mean_crp_raw = mean(crp_raw);
mean_das_raw = mean(das_raw);

var_crp_raw = std(crp_raw);
var_das_raw = std(das_raw);

mean_crp_wgt = mean(crp_wgt);
mean_das_wgt = mean(das_wgt);

var_crp_wgt = std(crp_wgt);
var_das_wgt = std(das_wgt);

y = [mean_crp_raw, mean_crp_wgt, mean_das_raw, mean_das_wgt]
er = [var_crp_raw, var_crp_wgt, var_das_raw, var_das_wgt]

colors = [0 1 0; 1 0 0; 0 0 1; 1 1 0]

figure(1)
hold on

ss = []

for c = 1:4
    e = errorbar(c, y(c), er(c), 'LineStyle','none', 'DisplayName','a');
    s = scatter(c, y(c));
    ss = [ss s];
    
    color = colors(c,:);
    
    e.Color = color * 0.4;
    s.MarkerEdgeColor = color * 0.75;
end    
hold off
lgd = legend(ss, 'CRP partial','CRP total','das28 partial','das28 total');

bins = 20

figure(2)
subplot(2,2,1)
hist(crp_raw, bins)
title('CRP partial')
subplot(2,2,2)
hist(crp_wgt, bins)
title('CRP total')
subplot(2,2,3)
hist(das_raw, bins)
title('das28 partial')
subplot(2,2,4)
hist(das_wgt, bins)
title('das28 total')

p1 = kruskalwallis([crp_raw; crp_wgt],[zeros(length(crp_raw),1); ones(length(crp_wgt),1)])

p2 = kruskalwallis([das_raw; das_wgt],[zeros(length(das_raw),1); ones(length(das_wgt),1)])