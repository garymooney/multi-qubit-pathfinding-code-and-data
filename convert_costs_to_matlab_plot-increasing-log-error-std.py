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

output_filename = 'matlab_plot_results_cost_compare_devices_error_std.m'
if os.path.exists(output_filename):
    os.remove(output_filename)
        
def write_line_to_file(text = '', output_filename=output_filename):
    try: 
        with open(output_filename, 'a') as file:
            file.write(text + "\n")

    except IOError:
        with open(output_filename, 'w') as file:
            file.write(text + "\n")
        
def generate_variable_string(variable_name, values):
    variable_string = str(variable_name) + " = ["
    add_space = False
    for value in values:
        if add_space == True:
            variable_string += " "
        else:
            add_space = True
        variable_string += str(value)
    variable_string += "];"
    return variable_string
     
results_list = []
results_list.append(load_dict_from_json("results-grid_8x8-increasing-error_stds.txt"))
results_list.append(load_dict_from_json("results-ibmq_rochester-increasing-error_stds.txt"))
results_list.append(load_dict_from_json("results-ibmq_paris-increasing-error_stds.txt"))
results_list.append(load_dict_from_json("results-rigetti_acorn-increasing-error_stds.txt"))
results_list.append(load_dict_from_json("results-poughkeepsie-increasing-error_stds.txt"))
results_list.append(load_dict_from_json("results-ibmq_melbourne-increasing-error_stds.txt"))

results_error_log10_stds = []
results_average_total_errors = []

for results in results_list:
    average_total_errors = []
    error_log10_std_set = set()
    for error_log10_std in results["error_log10_stds"]:
        error_log10_std_set.add(error_log10_std)
    error_log10_stds = sorted(list(error_log10_std_set))

    for error_log10_std in error_log10_stds:
        total_errors = []
        depths = []
        for i in range(len(results["runtimes"])):
            if results["error_log10_stds"][i] == error_log10_std:
                total_errors.append(results["total_errors"][i])
                if results["total_errors"][i] > 1:
                    print("Warning total_error > 1.")
                
        average_total_errors.append(statistics.mean(total_errors))
    
    results_error_log10_stds.append(error_log10_stds)
    results_average_total_errors.append(average_total_errors)
  
grid_error_log10_stds = results_error_log10_stds[0]
ibmq_rochester_error_log10_stds = results_error_log10_stds[1]
ibmq_paris_error_log10_stds = results_error_log10_stds[2]
rigetti_acorn_error_log10_stds = results_error_log10_stds[3]
ibmq_poughkeepsie_error_log10_stds = results_error_log10_stds[4]
ibmq_melbourne_error_log10_stds = results_error_log10_stds[5]

grid_average_total_errors = results_average_total_errors[0]
ibmq_rochester_average_total_errors = results_average_total_errors[1]
ibmq_paris_average_total_errors = results_average_total_errors[2]
rigetti_acorn_average_total_errors = results_average_total_errors[3]
ibmq_poughkeepsie_average_total_errors = results_average_total_errors[4]
ibmq_melbourne_average_total_errors = results_average_total_errors[5]

x_grid_error_log10_stds_string = generate_variable_string("x_grid_error_log10_stds", grid_error_log10_stds)
x_ibmq_rochester_error_log10_stds_string = generate_variable_string("x_ibmq_rochester_error_log10_stds", ibmq_rochester_error_log10_stds)
x_ibmq_paris_error_log10_stds_string = generate_variable_string("x_ibmq_paris_error_log10_stds", ibmq_paris_error_log10_stds)
x_rigetti_acorn_error_log10_stds_string = generate_variable_string("x_rigetti_acorn_error_log10_stds", rigetti_acorn_error_log10_stds)
x_ibmq_poughkeepsie_error_log10_stds_string = generate_variable_string("x_ibmq_poughkeepsie_error_log10_stds", ibmq_poughkeepsie_error_log10_stds)
x_ibmq_melbourne_error_log10_stds_string = generate_variable_string("x_ibmq_melbourne_error_log10_stds", ibmq_melbourne_error_log10_stds)

y_grid_average_total_errors_string = generate_variable_string("y_grid_average_total_errors", grid_average_total_errors)
y_ibmq_rochester_average_total_errors_string = generate_variable_string("y_ibmq_rochester_average_total_errors", ibmq_rochester_average_total_errors)
y_ibmq_paris_average_total_errors_string = generate_variable_string("y_ibmq_paris_average_total_errors", ibmq_paris_average_total_errors)
y_rigetti_acorn_average_total_errors_string = generate_variable_string("y_rigetti_acorn_average_total_errors", rigetti_acorn_average_total_errors)
y_ibmq_poughkeepsie_average_total_errors_string = generate_variable_string("y_ibmq_poughkeepsie_average_total_errors", ibmq_poughkeepsie_average_total_errors)
y_ibmq_melbourne_average_total_errors_string = generate_variable_string("y_ibmq_melbourne_average_total_errors", ibmq_melbourne_average_total_errors)

write_line_to_file(x_grid_error_log10_stds_string)
write_line_to_file(x_ibmq_rochester_error_log10_stds_string)
write_line_to_file(x_ibmq_paris_error_log10_stds_string)
write_line_to_file(x_rigetti_acorn_error_log10_stds_string)
write_line_to_file(x_ibmq_poughkeepsie_error_log10_stds_string)
write_line_to_file(x_ibmq_melbourne_error_log10_stds_string)

write_line_to_file(y_grid_average_total_errors_string)
write_line_to_file(y_ibmq_rochester_average_total_errors_string)
write_line_to_file(y_ibmq_paris_average_total_errors_string)
write_line_to_file(y_rigetti_acorn_average_total_errors_string)
write_line_to_file(y_ibmq_poughkeepsie_average_total_errors_string)
write_line_to_file(y_ibmq_melbourne_average_total_errors_string)

# Average total error by agent counts
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_grid_error_log10_stds, y_grid_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_rochester_error_log10_stds, y_ibmq_rochester_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_paris_error_log10_stds, y_ibmq_paris_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_rigetti_acorn_error_log10_stds, y_rigetti_acorn_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_poughkeepsie_error_log10_stds, y_ibmq_poughkeepsie_average_total_errors, 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_melbourne_error_log10_stds, y_ibmq_melbourne_average_total_errors, 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
#write_line_to_file("set(gca,'YScale','log');")
#write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
#write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q IBM Q Rochester', '27Q IBM Q Paris', '20Q Rigetti Acorn', '20Q IBM Q Poughkeepsie', '15Q IBM Q Melbourne', 'Location', 'Best');")
write_line_to_file("title('Cost of implementation');")
write_line_to_file("xlabel('log_{10} error std');")
write_line_to_file("ylabel('Average total error');")
#write_line_to_file("xlim([0 1])")
write_line_to_file("ylim([0 1]);")
