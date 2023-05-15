import sys
import numpy as np
import matplotlib.pyplot as plt

def calculate_hops(node1, node2, network_size):
    return abs(node1 - node2)

def calculate_hops_tor(node1, node2, network_size):
    # Adjusting the node numbering
    node1 -= 1
    node2 -= 1

    hops_forward = (node2 - node1) % network_size
    hops_backward = (node1 - node2) % network_size
    return min(hops_forward, hops_backward)

def calculate_hops_2D(node1, node2, rows, columns):
    row1, col1 = divmod(node1 - 1, columns)
    row2, col2 = divmod(node2 - 1, columns)
    row_diff = abs(row2 - row1)
    col_diff = abs(col2 - col1)

    return row_diff + col_diff

def calculate_hops_2D_torus(node1, node2, rows, columns):
    row1, col1 = divmod(node1 - 1, columns)
    row2, col2 = divmod(node2 - 1, columns)
    row_diff = min(abs(row2 - row1), rows - abs(row2 - row1))
    col_diff = min(abs(col2 - col1), columns - abs(col2 - col1))

    return row_diff + col_diff

# Check if the number of command line arguments is valid
if len(sys.argv) != 5:
    print("Usage: python torus_hops.py <number_of_nodes> <temporary_argument>")
    sys.exit(1)

# Parse command line arguments
network_size = int(sys.argv[1])
temporary_argument = sys.argv[2]
rows = int(sys.argv[3])
columns = int(sys.argv[4])

if (temporary_argument != '1D' and temporary_argument != '1Dtorus'):
    network_size=rows*columns
heatmap = np.zeros((network_size, network_size))
# Calculate number of hops for all node pairs
if(temporary_argument=='1D'):
    for node1 in range(1,network_size+1):
        for node2 in range(1,network_size+1):
            num_hops = calculate_hops(node1, node2, network_size)
            heatmap[node1-1][node2-1]=num_hops
            #print(f"Number of hops between {node1} and {node2} is: {num_hops}")
elif(temporary_argument=='1Dtorus'):
    for node1 in range(1,network_size+1):
        for node2 in range(1,network_size+1):
            num_hops = calculate_hops_tor(node1, node2, network_size)
            heatmap[node1-1][node2-1]=num_hops
            #print(f"Number of hops between {node1} and {node2} is: {num_hops}")
elif(temporary_argument=='2Dmesh'):
    network_size=rows*columns
    for node1 in range(1, rows*columns + 1):
        for node2 in range(1, rows*columns + 1):
            num_hops = calculate_hops_2D(node1, node2, rows, columns)
            heatmap[node1-1][node2-1]=num_hops
            #print(f"Number of hops between {node1} and {node2} is: {num_hops}")
elif(temporary_argument=='2Dmeshtorus'):
    for node1 in range(1, rows*columns + 1):
        for node2 in range(1, rows*columns + 1):
            num_hops = calculate_hops_2D_torus(node1, node2, rows, columns)
            heatmap[node1-1][node2-1]=num_hops
            #print(f"Number of hops between {node1} and {node2} in a {rows}x{columns} 2D mesh network with torus connectivity is: {num_hops}")

# Create the heatmap
n = network_size
# Create the heatmap
#plt.imshow(heatmap, cmap='Greens', interpolation='nearest')
plt.imshow(heatmap, cmap='Greens', interpolation='nearest', vmin=0, vmax=n)
# Add colorbar
cbar = plt.colorbar(ticks=np.arange(n+1))
cbar.ax.set_yticklabels(np.arange(n+1), fontsize=10)
cbar.set_label('Hops', rotation=270, labelpad=10)

# Set labels and title
plt.xticks(np.arange(n), np.arange(1, n+1))
plt.yticks(np.arange(n), np.arange(1, n+1))
plt.xlabel('Destination')
plt.ylabel('Source')
plt.title(f'NoC_MAP of '+temporary_argument)

# Show the plot
plt.show()
