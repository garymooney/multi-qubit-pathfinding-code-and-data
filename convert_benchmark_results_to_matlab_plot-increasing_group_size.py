import json
import os
import sys
import numpy as np
import statistics

def load_dict_from_json(filename):
    try:
        fh = open(filename, 'r')
    except FileNotFoundError:
        print("could not find:", filename)
        return None
    print("found:", filename)
        
    with open(filename) as f:
        return json.load(f)

output_filename = 'matlab_plot_results_benchmark.m'
if os.path.exists(output_filename):
    os.remove(output_filename)
        
def write_line_to_file(text = '', output_filename=output_filename):
    try: 
        with open(output_filename, 'a') as file:
            file.write(text + "\n")

    except IOError:
        with open(output_filename, 'w') as file:
            file.write(text + "\n")
        
     
results_list = []
results_list.append(load_dict_from_json("results-grid_8x8-increasing-group_counts.txt"))

results_sorted_list = []

for results in results_list:
    sorted_indices = np.argsort(results["runtimes"])
    results_sorted = {}
    results_sorted["experiment_layout"] = results["experiment_layout"]
    results_sorted["experiment_variables"] = results["experiment_variables"]
    results_sorted["runtimes"] = []
    results_sorted["total_costs"] = []
    results_sorted["total_errors"] = []
    results_sorted["node_counts"] = []
    results_sorted["agent_counts"] = []
    results_sorted["split_counts"] = []
    results_sorted["dead_node_counts"] = []
    results_sorted["error_means"] = []
    results_sorted["error_log10_stds"] = []
    
    for index in sorted_indices:
        results_sorted["runtimes"].append(results["runtimes"][index])
        results_sorted["total_costs"].append(results["total_costs"][index])
        results_sorted["total_errors"].append(results["total_errors"][index])
        results_sorted["node_counts"].append(results["node_counts"][index])
        results_sorted["agent_counts"].append(results["agent_counts"][index])
        results_sorted["split_counts"].append(results["split_counts"][index])
        results_sorted["dead_node_counts"].append(results["dead_node_counts"][index])
        results_sorted["error_means"].append(results["error_means"][index])
        results_sorted["error_log10_stds"].append(results["error_log10_stds"][index])
    
    results_sorted_list.append(results_sorted)
    
results_average_runtimes = []
results_average_total_errors = []
split_counts_set = set()

for split_count in results["split_counts"]:
    split_counts_set.add(split_count)
results_split_counts = sorted(list(split_counts_set))

for split_count in results_split_counts:
    runtimes = []
    total_errors = []
    for i in range(len(results_list[0]["runtimes"])):
        if results_list[0]["split_counts"][i] == split_count:
            runtimes.append(results_list[0]["runtimes"][i])
            total_errors.append(results_list[0]["total_errors"][i])
            
    results_average_runtimes.append(statistics.mean(runtimes))
    results_average_total_errors.append(statistics.mean(total_errors))
        



results_sorted_list.append(results_sorted)
        
grid_increasing_group_benchmark_results_sorted = results_sorted_list[0]

x_instances_increasing_group_string = "x_instances_independent = ["
add_space = False
for i in range(len(grid_increasing_group_benchmark_results_sorted["runtimes"])):
    if add_space == True:
        x_instances_increasing_group_string += " "
    else:
        add_space = True
    x_instances_increasing_group_string += str(i+1)
x_instances_increasing_group_string += "];"

y_runtimes_increasing_group_string = "y_runtimes_increasing_group = ["
add_space = False
for runtime in grid_increasing_group_benchmark_results_sorted["runtimes"]:
    if add_space == True:
        y_runtimes_increasing_group_string += " "
    else:
        add_space = True
    y_runtimes_increasing_group_string += str(runtime)
y_runtimes_increasing_group_string += "];"

x_split_counts_string = "x_split_counts = ["
add_space = False
for split_count in results_split_counts:
    if add_space == True:
        x_split_counts_string += " "
    else:
        add_space = True
    x_split_counts_string += str(split_count)
x_split_counts_string += "];"

y_runtimes_split_counts_string = "y_runtimes_split_counts = ["
add_space = False
for runtime in results_average_runtimes:
    if add_space == True:
        y_runtimes_split_counts_string += " "
    else:
        add_space = True
    y_runtimes_split_counts_string += str(runtime)
y_runtimes_split_counts_string += "];"

y_total_errors_string = "y_total_errors = ["
add_space = False
for total_error in results_average_total_errors:
    if add_space == True:
        y_total_errors_string += " "
    else:
        add_space = True
    y_total_errors_string += str(total_error)
y_total_errors_string += "];"

write_line_to_file("")
write_line_to_file(x_instances_increasing_group_string)
write_line_to_file(y_runtimes_increasing_group_string)

write_line_to_file(x_split_counts_string)
write_line_to_file(y_runtimes_split_counts_string)
write_line_to_file(y_total_errors_string)

write_line_to_file("y_runtimes_increasing_group = y_runtimes_increasing_group./1000")
write_line_to_file("y_runtimes_split_counts = y_runtimes_split_counts./1000")

write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_instances_independent, y_runtimes_increasing_group, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("set(gca,'YScale','log');")
write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
#write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Grid 8x8 runtimes with increasing group sizes');")
write_line_to_file("xlabel('Instance');")
write_line_to_file("ylabel('Runtime (seconds)');")
write_line_to_file("xlim([0 " + str(len(grid_increasing_group_benchmark_results_sorted["runtimes"])+1) + "])")

write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_split_counts, y_runtimes_split_counts, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
write_line_to_file("set(gca,'YScale','log');")
write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
#write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Grid 8x8 average runtimes with increasing group sizes');")
write_line_to_file("xlabel('Group sizes');")
write_line_to_file("ylabel('Runtime (seconds)');")
write_line_to_file("xlim([0 " + str(len(results_split_counts)+1) + "])")
write_line_to_file("ylim([0 1000])")

write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_split_counts, y_total_errors, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
#write_line_to_file("set(gca,'YScale','log');")
#write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
#write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
#write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Grid 8x8 error cost with increasing group sizes');")
write_line_to_file("xlabel('Group sizes');")
write_line_to_file("ylabel('Total error cost');")
write_line_to_file("xlim([0 " + str(len(results_split_counts)+1) + "])")
write_line_to_file("ylim([0 1])")