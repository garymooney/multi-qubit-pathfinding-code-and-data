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

output_filename = 'matlab_plot_results_timeouts.m'
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
results_list.append(load_dict_from_json("results-grid-8x8-increasing-independent-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_rochester-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_paris-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-rigetti_acorn-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_poughkeepsie-increasing-individual-agents.txt"))
results_list.append(load_dict_from_json("results-ibmq_melbourne-increasing-individual-agents.txt"))

x_axis_label = "agent_counts"
y_axis_label = "timeout_counts"
x_axis_label_final = "Agent count"
y_axis_label_final = "Timeout count"
results_agent_counts = []
results_timeouts = []

for results in results_list:
    
    max_agents = max(results["agent_counts"])
    agent_counts = []
    timeouts = []
    for i in range(1,max_agents+1):
        timeouts.append(10 - results["agent_counts"].count(i))
        agent_counts.append(i)
    
    results_agent_counts.append(agent_counts)
    results_timeouts.append(timeouts)
    

grid_x_axis = results_agent_counts[0]
ibmq_rochester_x_axis = results_agent_counts[1]
ibmq_paris_x_axis = results_agent_counts[2]
rigetti_acorn_x_axis = results_agent_counts[3]
ibmq_poughkeepsie_x_axis = results_agent_counts[4]
ibmq_melbourne_x_axis = results_agent_counts[5]

grid_y_axis = results_timeouts[0]
ibmq_rochester_y_axis = results_timeouts[1]
ibmq_paris_y_axis = results_timeouts[2]
rigetti_acorn_y_axis = results_timeouts[3]
ibmq_poughkeepsie_y_axis = results_timeouts[4]
ibmq_melbourne_y_axis = results_timeouts[5]

x_grid_string = generate_variable_string("x_grid_" + x_axis_label, grid_x_axis)
x_ibmq_rochester_string = generate_variable_string("x_ibmq_rochester_" + x_axis_label, ibmq_rochester_x_axis)
x_ibmq_paris_string = generate_variable_string("x_ibmq_paris_" + x_axis_label, ibmq_paris_x_axis)
x_rigetti_acorn_string = generate_variable_string("x_rigetti_acorn_" + x_axis_label, rigetti_acorn_x_axis)
x_ibmq_poughkeepsie_string = generate_variable_string("x_ibmq_poughkeepsie_" + x_axis_label, ibmq_poughkeepsie_x_axis)
x_ibmq_melbourne_string = generate_variable_string("x_ibmq_melbourne_" + x_axis_label, ibmq_melbourne_x_axis)

y_grid_string = generate_variable_string("y_grid_" + y_axis_label, grid_y_axis)
y_ibmq_rochester_string = generate_variable_string("y_ibmq_rochester_" + y_axis_label, ibmq_rochester_y_axis)
y_ibmq_paris_string = generate_variable_string("y_ibmq_paris_" + y_axis_label, ibmq_paris_y_axis)
y_rigetti_acorn_string = generate_variable_string("y_rigetti_acorn_" + y_axis_label, rigetti_acorn_y_axis)
y_ibmq_poughkeepsie_string = generate_variable_string("y_ibmq_poughkeepsie_" + y_axis_label, ibmq_poughkeepsie_y_axis)
y_ibmq_melbourne_string = generate_variable_string("y_ibmq_melbourne_" + y_axis_label, ibmq_melbourne_y_axis)

write_line_to_file(x_grid_string)
write_line_to_file(x_ibmq_rochester_string)
write_line_to_file(x_ibmq_paris_string)
write_line_to_file(x_rigetti_acorn_string)
write_line_to_file(x_ibmq_poughkeepsie_string)
write_line_to_file(x_ibmq_melbourne_string)

write_line_to_file(y_grid_string)
write_line_to_file(y_ibmq_rochester_string)
write_line_to_file(y_ibmq_paris_string)
write_line_to_file(y_rigetti_acorn_string)
write_line_to_file(y_ibmq_poughkeepsie_string)
write_line_to_file(y_ibmq_melbourne_string)

# Average total error by agent counts
write_line_to_file("")
write_line_to_file("figure;")
write_line_to_file("")
write_line_to_file("plot(x_grid_" + x_axis_label + ", y_grid_" + y_axis_label + ", 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_rochester_" + x_axis_label + ", y_ibmq_rochester_" + y_axis_label + ", 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_paris_" + x_axis_label + ", y_ibmq_paris_" + y_axis_label + ", 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_rigetti_acorn_" + x_axis_label + ", y_rigetti_acorn_" + y_axis_label + ", 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_poughkeepsie_" + x_axis_label + ", y_ibmq_poughkeepsie_" + y_axis_label + ", 'LineWidth', 2);")
write_line_to_file("hold on;")
write_line_to_file("plot(x_ibmq_melbourne_" + x_axis_label + ", y_ibmq_melbourne_" + y_axis_label + ", 'LineWidth', 2);")

#write_line_to_file("scatter(1:length(x_values), y_values_max_with_measurement_correction_custom, 65, '.');")
write_line_to_file("grid on")
write_line_to_file("set(gcf,'color','w');")
#write_line_to_file("set(gca,'YScale','log');")
#write_line_to_file("set(gca,'YTick',[0.001, 0.01, 0.1, 1, 10, 100, 1000],...")
#write_line_to_file("        'YTickLabel',{'0.001', '0.01', '0.1', '1', '10', '100', '1000'})")
#write_line_to_file("set(gca,'XTick',1:length(x_values), 'XTickLabel', x_values);")
#write_line_to_file("xtickangle(90);")
write_line_to_file("legend('64Q Grid 8x8', '53Q \\it{ibmq\\_rochester}', '27Q \\it{ibmq\\_paris}', '20Q Rigetti Acorn', '20Q \\it{ibmq\\_poughkeepsie}', '15Q \\it{ibmq\\_16\\_melbourne}', 'Location', 'Best');")
#write_line_to_file("title('Cost of implementation');")
write_line_to_file("xlabel('" + x_axis_label_final + "');")
write_line_to_file("ylabel('" + y_axis_label_final + "');")
#write_line_to_file("xlim([0 1])")
#write_line_to_file("ylim([0 1]);")
