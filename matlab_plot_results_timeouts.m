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

figure;

plot(x_grid_agent_counts, y_grid_timeout_counts, 'LineWidth', 2);
hold on;
plot(x_ibmq_rochester_agent_counts, y_ibmq_rochester_timeout_counts, 'LineWidth', 2);
hold on;
plot(x_ibmq_paris_agent_counts, y_ibmq_paris_timeout_counts, 'LineWidth', 2);
hold on;
plot(x_rigetti_acorn_agent_counts, y_rigetti_acorn_timeout_counts, 'LineWidth', 2);
hold on;
plot(x_ibmq_poughkeepsie_agent_counts, y_ibmq_poughkeepsie_timeout_counts, 'LineWidth', 2);
hold on;
plot(x_ibmq_melbourne_agent_counts, y_ibmq_melbourne_timeout_counts, 'LineWidth', 2);
grid on
set(gcf,'color','w');
legend('64Q Grid 8x8', '53Q \it{ibmq\_rochester}', '27Q \it{ibmq\_paris}', '20Q Rigetti Acorn', '20Q \it{ibmq\_poughkeepsie}', '15Q \it{ibmq\_16\_melbourne}', 'Location', 'Best');
xlabel('Agent count');
ylabel('Timeout count');
