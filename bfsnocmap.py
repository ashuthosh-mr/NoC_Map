import sys
import numpy as np
import matplotlib.pyplot as plt

#function definitions

#binary tree
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def construct_tree(node_values, index):
    if index < len(node_values):
        value = node_values[index]
        if value is None:
            return None

        node = Node(value)
        node.left = construct_tree(node_values, 2 * index + 1)
        node.right = construct_tree(node_values, 2 * index + 2)
        return node

def find_node(root, value):
    if root is None or root.data == value:
        return root

    left_subtree = find_node(root.left, value)
    if left_subtree is not None:
        return left_subtree

    right_subtree = find_node(root.right, value)
    return right_subtree

def find_distance(root, node1, node2):
    lca = find_lca(root, node1, node2)
    distance_node1 = find_distance_from_node(lca, node1, 0)
    distance_node2 = find_distance_from_node(lca, node2, 0)
    return distance_node1 + distance_node2

def find_lca(root, node1, node2):
    if root is None:
        return None

    if root == node1 or root == node2:
        return root

    left_lca = find_lca(root.left, node1, node2)
    right_lca = find_lca(root.right, node1, node2)

    if left_lca and right_lca:
        return root

    return left_lca if left_lca is not None else right_lca

def find_distance_from_node(node, target, distance):
    if node is None:
        return -1

    if node == target:
        return distance

    left_distance = find_distance_from_node(node.left, target, distance + 1)
    if left_distance != -1:
        return left_distance

    right_distance = find_distance_from_node(node.right, target, distance + 1)
    return right_distance


####################################################do not touch
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

def calculate_hops_alltoall(node1, node2,network_size):
    if(node1==node2):
        return node1-node2
    else:
        return 1

    return row_diff + col_diff
def plotheat(temporary_argument, heatmap, n, kind):#subplot_row, subplot_col, subplot_index,kind):
    network_size = n
    #alias_i = 0
    #alias_j = 0
    maximum_value = np.max(heatmap)
    #placing = [0]*n
    # Create the heatmap
    plt.imshow(heatmap, cmap='Greens', interpolation='nearest', vmin=0, vmax=maximum_value)

    # Add colorbar
#    cbar = plt.colorbar(ticks=np.arange(network_size+1))
#    cbar.ax.set_yticklabels(np.arange(network_size+1), fontsize=10)
#    cbar.set_label('Hops', rotation=270, labelpad=10)

    # Set labels and title
    plt.xticks(np.arange(network_size), np.arange(1, network_size+1))
    plt.yticks(np.arange(network_size), np.arange(1, network_size+1))
    plt.xlabel('Destination')
    plt.ylabel('Source')
    plt.title('NoC_MAP of ' + temporary_argument+' and '+kind +','+ part)
    for i in range(network_size):
        for j in range(network_size):
            plt.text(j, i, str(int(heatmap[i, j])), color='black', ha='center', va='center',fontsize=8)
    #plt.suptitle('This is the plot of '+kind)
    plt.savefig('./plots/'+temporary_argument+'.png')
    plt.close()


# Check if the number of command line arguments is valid
if len(sys.argv) != 7:
    print("Usage: python bfsnocmap.py <number_of_nodes> <graph> <rows> <columns> <map,communication_cost,cfactor>, <partition>")
    sys.exit(1)

# Parse command line arguments
n = int(sys.argv[1])
graph = sys.argv[2]
rows = int(sys.argv[3])
columns = int(sys.argv[4])
kind = sys.argv[5]
part = sys.argv[6]
import importlib
module_name = 'data_' + graph
importlib.invalidate_caches()  # Optional: Clear import caches if needed
data_module = importlib.import_module(module_name)
h_graph_edges = data_module.h_graph_edges
h_graph_nodes = data_module.h_graph_nodes

#take input file from metis
i=0
a=0;
d=0;
V=0;
if part=="cut":
    string='./partitions/edge/tometis'
else:
    string='./partitions/tometis'
with open(string+graph+'.part.'+str(n)) as f3:
    for line in f3:
        Type = line.split()
        V=V+1
#generate howns
hown = np.zeros((n,V), dtype=int)
send_local = np.zeros((n,n), dtype=int)
send_total = [0]*n
receive_total = [0]*n
'''
while i<n:
    a=0
    d=0
    with open(string+graph+'.part.'+str(n)) as f3:
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
'''

p = 0
for j in range(0,795):
    hown[0][j]=p
    p=p+1
for j in range(795,795+834):
    hown[0][j]=-2
for j in range(795+834,795+834+832):
    hown[0][j]=-3
for j in range(795+834+832,V):
    hown[0][j]=-4
print(p)

p=0
for j in range(0,795):
    hown[1][j]=-1
for j in range(795,795+834):
    hown[1][j]=p
    p=p+1
for j in range(795+834,795+834+832):
    hown[1][j]=-3
for j in range(795+834+832,V):
    hown[1][j]=-4
print(p)
p=0
for j in range(0,795):
    hown[2][j]=-1
for j in range(795,795+834):
    hown[2][j]=-2
for j in range(795+834,795+834+832):
    hown[2][j]=p
    p=p+1
for j in range(795+834+832,V):
    hown[2][j]=-4
print(p)
    
p=0

for j in range(0,795):
    hown[3][j]=-1
for j in range(795,795+834):
    hown[3][j]=-2
for j in range(795+834,795+834+832):
    hown[3][j]=-3
for j in range(795+834+832,V):
    hown[3][j]=p
    p=p+1
print(p)
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
#cfactor = [0.0]*n
cfactor = np.zeros((n,n), dtype=float)

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
#for i in range(len(send_total)):
#    cfactor[i]=round((inter_communication[i]/intra_communication[i]),2)

placing = [0]*n
	
    # Open the file in read mode
with open('placing.in', 'r') as file:
    for line in file:
        # Remove leading/trailing whitespaces and newline characters
        line = line.strip()
        # Split the line by '->' to extract the values
        values = line.split('->')
            # Extract the values
        value1 = int(values[0].strip())
        value2 = int(values[1].strip())
        placing[value1]=value2
#debug
#for i in range (0,n-1):
#    temp2=temp2+send_local[15][i]
#print(temp2)
#send_total is the main thing to be plotted

network_size = n
network_size=rows*columns
for i in range(0,network_size):
    for j in range(0,network_size):
        if(send_local[i][j]!=0):
            cfactor[i][j]=round((send_local[i][i]/send_local[i][j]),2)
heatmap = np.zeros((network_size, network_size))
# Open the file in read mode
# Calculate number of hops for all node pairs
temporary_argument = 'alltoall'
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops_alltoall(placing[node1-1], placing[node2-1], network_size)
        heatmap[node1-1][node2-1]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
#plt.subplot(2, 3, 1)
#plotheat(temporary_argument, heatmap, n,2,3,1,kind)
plotheat(temporary_argument, heatmap, n,kind)

temporary_argument = '1D'
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops(placing[node1-1], placing[node2-1], network_size)
        heatmap[node1-1][node2-1]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
#plt.subplot(1, 1, 1)
#plotheat(temporary_argument, heatmap, n,1,1,1,kind)
plotheat(temporary_argument, heatmap, n,kind)

temporary_argument = '1Dtorus'
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops_tor(placing[node1-1], placing[node2-1], network_size)      
        heatmap[node1-1][node2-1]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
#plt.subplot(2, 3, 3)
#plotheat(temporary_argument, heatmap, n,2,3,3,kind)
plotheat(temporary_argument, heatmap, n,kind)

temporary_argument = '2Dmesh'
for node1 in range(1, rows*columns + 1):
    for node2 in range(1, rows*columns + 1):
        num_hops = calculate_hops_2D(placing[node1-1], placing[node2-1], rows, columns)
        heatmap[node1-1][node2-1]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
#plt.subplot(2, 3, 4)
#plotheat(temporary_argument, heatmap, n,2,3,4,kind)
plotheat(temporary_argument, heatmap, n,kind)

temporary_argument = '2Dmeshtorus'
for node1 in range(1, rows*columns + 1):
    for node2 in range(1, rows*columns + 1):
        num_hops = calculate_hops_2D_torus(placing[node1-1], placing[node2-1], rows, columns)
        heatmap[node1-1][node2-1]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
#plt.subplot(2, 3, 5)
#plotheat(temporary_argument, heatmap, n,2,3,5,kind)
# Adjust the spacing between subplots
plotheat(temporary_argument, heatmap, n,kind)


# Example usage
node_values = [0]*n

#for i in range(0,n):
#    alias_i = placing[i]
#        for j in range(0,n):
#            alias_j = placing[j]
#            heatmap[i][j]= heatmap1[alias_i][alias_j]
for i in range(0,n):
    node_values[i]=i+1
root = construct_tree(node_values, 0)
temporary_argument = 'binarytree'
for node1 in range(0, n):
    for node2 in range(0, n):
        node3 = find_node(root, node1+1)
        node4 = find_node(root, node2+1)
        num_hops = find_distance(root,node3,node4)
        heatmap[placing[node1]][placing[node2]]=num_hops
if kind=='communication_cost':
    for i in range(0,n):
        for j in range(0,n):
            if(send_local[i][j]!=0):
                heatmap[i][j]=heatmap[i][j]*send_local[i][j]
elif kind=='cfactor':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=cfactor[i][j]
elif kind=='partition':
    for i in range(0,n):
        for j in range(0,n):
            heatmap[i][j]=send_local[i][j]
plotheat(temporary_argument, heatmap, n, kind)

# Find the distance between nodes 4 and 7
#plt.tight_layout()
# Show the plot
#plt.show()
