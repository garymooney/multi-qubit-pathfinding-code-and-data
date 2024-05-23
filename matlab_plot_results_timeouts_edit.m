x_grid_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39];
x_ibmq_rochester_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48];
x_ibmq_paris_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27];
x_rigetti_acorn_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
x_ibmq_poughkeepsie_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
x_ibmq_melbourne_agent_counts = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
y_grid_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 6 2 0 4 5 7 7 10 10 7 8];
y_ibmq_rochester_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 1 0 5 3 3 6 5 5 9 5 10 6 6 9 8 9 10 9 9 9];
y_ibmq_paris_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
y_rigetti_acorn_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
y_ibmq_poughkeepsie_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
y_ibmq_melbourne_timeout_counts = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

x_width=5;
y_width=4;

figure;

scatter(x_grid_agent_counts, y_grid_timeout_counts-0.15, 200, '.');
hold on;
scatter(x_ibmq_rochester_agent_counts, y_ibmq_rochester_timeout_counts+0.06-0.15, 200, '.');
hold on;
scatter(x_ibmq_paris_agent_counts, y_ibmq_paris_timeout_counts+0.12-0.15, 200, '.');
hold on;
scatter(x_rigetti_acorn_agent_counts, y_rigetti_acorn_timeout_counts+0.18-0.15, 200, '.');
hold on;
scatter(x_ibmq_poughkeepsie_agent_counts, y_ibmq_poughkeepsie_timeout_counts+0.24-0.15, 200, '.');
hold on;
scatter(x_ibmq_melbourne_agent_counts, y_ibmq_melbourne_timeout_counts+0.30-0.15, 200, '.');
grid on
set(gcf,'color','w');
[h,icons] = legend('64q Grid 8x8', '53q ibmq\_rochester', '27q ibmq\_paris', '20q Rigetti Acorn', '20q ibmq\_poughkeepsie', '15q ibmq\_16\_melbourne', 'Location', 'northwest');
% Find the 'line' objects
icons = findobj(icons,'Type','patch');
% Find lines that use a marker
icons = findobj(icons,'Marker','none','-xor');

set(icons, 'MarkerSize', 15);
xlabel('Logical qubit count');
ylabel('Timeout count');
xlim([1 50])
ylim([-0.25 10.25])
set(gca,'MinorGridLineStyle',':');
set(gcf, 'PaperSize', [x_width y_width]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [x_width y_width]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
set(gcf, 'renderer', 'painters');
saveas(gcf,'output/timeouts_vs_agent_counts.png', 'png');
print(gcf, '-dpdf', 'output/timeouts_vs_agent_counts.pdf');
saveas(gcf,'output/timeouts_vs_agent_counts.fig', 'fig');

