import sys
import numpy as np
import matplotlib.pyplot as plt
from data import h_graph_edges,h_graph_nodes

#function definitions


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

def plotheat(temporary_argument, heatmap, n):
    network_size = n

    # Create the heatmap
    plt.imshow(heatmap, cmap='Greens', interpolation='nearest', vmin=0, vmax=network_size)

    # Add colorbar
    cbar = plt.colorbar(ticks=np.arange(network_size+1))
    cbar.ax.set_yticklabels(np.arange(network_size+1), fontsize=10)
    cbar.set_label('Hops', rotation=270, labelpad=10)

    # Set labels and title
    plt.xticks(np.arange(network_size), np.arange(1, network_size+1))
    plt.yticks(np.arange(network_size), np.arange(1, network_size+1))
    plt.xlabel('Destination')
    plt.ylabel('Source')
    plt.title('NoC_MAP of ' + temporary_argument)

    plt.show(block=False)

# Check if the number of command line arguments is valid
if len(sys.argv) != 6:
    print("Usage: python communication.py <number_of_nodes> <graph> <number_of_vertices> <rows> <columns>")
    sys.exit(1)

# Parse command line arguments
n = int(sys.argv[1])
graph = sys.argv[2]
V = int(sys.argv[3])
rows = int(sys.argv[4])
columns = int(sys.argv[5])

#take input file from metis
i=0
a=0;
d=0;
#generate howns
hown = np.zeros((n,V), dtype=int)
send_local = np.zeros((n,n), dtype=int)
send_total = [0]*n
receive_total = [0]*n
while i<n:
    a=0
    d=0
    with open('tometis'+graph+'.part.'+str(n)) as f3:
        for line in f3:
            Type = line.split()
            if(Type[0]==str(i)):
                hown[i][d]=a
                a=a+1
            else:
                temp=int(Type[0])+1
                hown[i][d]=-1*temp
            d=d+1
    i=i+1
#bfs
h_graph_active = []
h_updating_graph_active = []
h_cost = [-1]*V
count=1
count1=1
j=0
tid=0
id=0
stop=0
k=0
iter=0
h_graph_active.append(0)
while count:
    count=0
    for j in range (0,count1):
        tid=h_graph_active[j]
        for k in range(h_graph_nodes[tid],h_graph_nodes[tid+1]):
            id=h_graph_edges[k]
            i=0
            while i<n:
                if(hown[i][tid]>=0):
                    if(hown[i][id]<0):
                        temp1=abs(hown[i][id])-1
                        send_local[i][temp1]=send_local[i][temp1]+1
                    else:
                        send_local[i][i]=send_local[i][i]+1
                i=i+1
            if(h_cost[id]<0):
                h_cost[id]=iter
                h_updating_graph_active.append(id)
                count=count+1
    count1=count
    iter=iter+1
    h_graph_active=h_updating_graph_active.copy()
    h_updating_graph_active.clear()
h_graph_active.clear()
h_updating_graph_active.clear()
print(iter)
temp2=0
inter_communication = [0]*n
intra_communication = [0]*n
cfactor = [0.0]*n
for i in range (0,n):
    for j in range (0,n):
        if(i!=j):
            send_total[i]=send_total[i]+send_local[i][j]
        if(i==j):
            intra_communication[i]=intra_communication[i]+send_local[i][i]
for i in range (0,n):
    for j in range (0,n):
        if(i!=j):
            receive_total[i]=receive_total[i]+send_local[j][i]

for i in range(len(send_total)):
    inter_communication[i] = send_total[i] + receive_total[i]
for i in range(len(send_total)):
    cfactor[i]=round((inter_communication[i]/intra_communication[i]),2)

#debug
for i in range (0,n-1):
    temp2=temp2+send_local[15][i]
print(temp2)
#send_total is the main thing to be plotted

network_size = n
network_size=rows*columns
heatmap = np.zeros((network_size, network_size))
# Calculate number of hops for all node pairs
temporary_argument = '1D'
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops(node1, node2, network_size)
        heatmap[node1-1][node2-1]=num_hops
plotheat(temporary_argument, heatmap, n)
temporary_argument = '1Dtorus'
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops_tor(node1, node2, network_size)
        heatmap[node1-1][node2-1]=num_hops
plotheat(temporary_argument, heatmap, n)
temporary_argument = '2Dmesh'
for node1 in range(1, rows*columns + 1):
    for node2 in range(1, rows*columns + 1):
        num_hops = calculate_hops_2D(node1, node2, rows, columns)
        heatmap[node1-1][node2-1]=num_hops
plotheat(temporary_argument, heatmap, n)
temporary_argument = '2Dmeshtorus'
for node1 in range(1, rows*columns + 1):
    for node2 in range(1, rows*columns + 1):
        num_hops = calculate_hops_2D_torus(node1, node2, rows, columns)
        heatmap[node1-1][node2-1]=num_hops
plotheat(temporary_argument, heatmap, n)