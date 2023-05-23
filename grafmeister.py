import sys
import numpy as np
import matplotlib.pyplot as plt

#function definitions
def generate_factor_pairs(number):
    factor_pairs = []

    # Iterate from 1 to the square root of the number
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            factor1 = i
            factor2 = number // i
            factor_pairs.append((factor1, factor2))

    return factor_pairs

def find_max_factor_pairs(number):
    factor_pairs = generate_factor_pairs(number)
    max_product = 0
    max_factor_pair = None
    maxstart = 0
    if(len(factor_pairs)==1):
        max_factor_pair = factor_pairs[0]
        return max_factor_pair
    else:
        for factor_pair in factor_pairs:
            if factor_pair[0] > maxstart:
                maxstart = factor_pair[0]
                product = factor_pair[0] * factor_pair[1]
                if product >= max_product:
                    max_product = product
                    max_factor_pair = factor_pair
        return max_factor_pair

def find_ideal_cores(speedup,tag):
    x = range(1, len(speedup) + 1)

    plt.plot(x, speedup, marker='o')
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup')
    if(tag==0):
        plt.title('Speedup vs Number of Cores')
    else:
        plt.title('NoC Speedup vs Number of Cores')
    plt.grid(True)
    plt.show()

    # Calculate the derivative of the speedup curve
    derivatives = np.gradient(speedup)

    # Find the index of the minimum derivative value
    max_derivative_index = np.argmax(derivatives)

    # Get the corresponding number of cores
    max_derivative_cores = max_derivative_index + 1

    return max_derivative_cores

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

if len(sys.argv) != 4:
    print("Usage: python graphmeister.py <number_of_nodes> <graph> <noc> ")
    sys.exit(1)

n = int(sys.argv[1])
#rows = int(sys.argv[2])
#columns = int(sys.argv[3])
graph = sys.argv[2]
noc = sys.argv[3]

import importlib
module_name = 'data_' + graph
importlib.invalidate_caches()  # Optional: Clear import caches if needed
data_module = importlib.import_module(module_name)
h_graph_edges = data_module.h_graph_edges
h_graph_nodes = data_module.h_graph_nodes

#vertices and edges retrieval
V=len(h_graph_nodes)-1
E=len(h_graph_edges)

#single core latency
single_latency = (V*146)+((E-V)*40)
#print('Single core latency: ', single_latency)
core_latency = [0]*n
speedup = [0.0]*n
core_latency[0] = single_latency
speedup[0]=1
max_core = n
core_latency_noc = [0]*n
speedup_noc = [0.0]*n
speedup_noc[0]=1
core_latency_noc[0] = single_latency

#make hown
hown = [0]*V
n = 2
while n<max_core + 1:
    i=0
    a=0
    d=0
    count=0
    with open('./partitions/tometis'+graph+'.part.'+str(n)) as f3:
        for line in f3:
            Type = line.split()
            hown[d]= int(Type[0])
            d=d+1
    #count vertices, edges and edge crossings
    vcount = [0]*n
    ecount = [0]*n

    while i<n:
        vcount[i] = hown.count(i)
        i=i+1
    ##########vcount is done.
    core=0
    for i in range (0,V):
        count=h_graph_nodes[i+1]-h_graph_nodes[i]
        core=hown[i]
        ecount[core]=ecount[core]+count
    #########ecount is done
    send_local = np.zeros((n,n), dtype=int)
    x=0
    y=0
    for i in range (0,V):
        x=hown[i]
        for j in range(h_graph_nodes[i],h_graph_nodes[i+1]):
            id=h_graph_edges[j]
            y=hown[id]
            send_local[x][y]=send_local[x][y]+1
    crosscom = [0]*n
    for j in range(0,n):
        for i in range(0,n):
            if(i!=j):
                crosscom[j]=crosscom[j]+send_local[j][i]
    ###########cross communication done
    ##model begins here
    traversal = [0]*n
    sync = [0]*n
    communication = [0]*n
    total = [0]*n
    #constants go here
    c1 = 156 #active vertex
    c2 = 40 #edges - active vertex
    c3 = 82
    c4 = 0
    c5 = 0
    iter = 8
    arbit_latency = ((n-1)*3)+((n-1)*2)+2
    ############
    #Traversal latency:
    for i in range(0,n):
        traversal[i]=(vcount[i]*c1)+((ecount[i]-vcount[i])*c2)
    #############Sync latency
    for i in range(0,n):
        temp1=(max(ecount)-min(ecount))/ecount[i]
        temp2=(vcount[i]*c1)+((ecount[i]-vcount[i])*c2)+(arbit_latency*crosscom[i])
        sync[i]=int(temp1*temp2)+((2*iter)*(((n-1)*arbit_latency)+arbit_latency+arbit_latency))
    ############communication latency
    for i in range(0,n):
        temp1=(crosscom[i]*(arbit_latency+arbit_latency+c3))
        temp2=(n-1)*iter*(arbit_latency+arbit_latency+8)
        communication[i]=temp1+temp2
    ###########total latency
    for i in range(0,n):
        total[i] = traversal[i] + communication[i] + sync[i]
    speedup[n-1] = round((single_latency / max(total)), 2)
    core_latency[n-1] = max(total)
#####NOC based
#print('Done')
#print(core_latency)
#print(speedup)
    max_factor_pair = find_max_factor_pairs(n)
#    print("Maximum Factor Pair:", max_factor_pair)
    rows = max_factor_pair[0]
    columns = max_factor_pair [1]
    send_noc = np.zeros((n,n), dtype=int)
    deg = 0
    if noc == '1D':
        deg = 2
    elif noc == '1Dtorus':
        deg = 2
    elif noc == '2Dmesh':
        deg  = 4
    elif noc == '2Dmeshtorus':
        deg  = 4
    elif noc == 'alltoall':
        sys.exit(1)

    arbit_latency_noc = ((deg-1)*3)+((deg-1)*2)+2
    network_size = n
    network_size=rows*columns

    # Calculate number of hops for all node pairs
    for node1 in range(1,network_size+1):
        for node2 in range(1,network_size+1):
            if noc == '1D':
                num_hops = calculate_hops(node1, node2, network_size)
            elif noc =='1Dtorus':
                num_hops = calculate_hops_tor(node1, node2, network_size)
            elif noc == '2Dmesh':
                num_hops = calculate_hops_2D(node1, node2, rows, columns)
            elif noc == '2Dmeshtorus':
                num_hops = calculate_hops_2D_torus(node1, node2, rows,columns)
            if(node1!=node2):
                send_noc[node1-1][node2-1]=num_hops*arbit_latency_noc
            else:
                send_noc[node1-1][node2-1]=1
    # noc based accumulation
    crosscom_noc = [0]*n
    for j in range(0,n):
        for i in range(0,n):
            if(i!=j):
                crosscom_noc[j]=crosscom_noc[j]+send_noc[j][i]
    ####sync
    temp3 = np.max(send_noc)
    #debug print(temp3)
    for i in range(0,n):
        temp1=(max(ecount)-min(ecount))/ecount[i]
        temp2=(vcount[i]*c1)+((ecount[i]-vcount[i])*c2)+(crosscom_noc[i])
        sync[i]=int(temp1*temp2)+((2*iter)*(((n-1)*temp3)+temp3+8))
    ############communication latency
    for i in range(0,n):
        temp1=(crosscom_noc[i]*(2+c3))
        temp2=(n-1)*iter*(temp3+temp3+8)
        communication[i]=temp1+temp2
    ###########total latency
    for i in range(0,n):
        total[i] = traversal[i] + communication[i] + sync[i]
    speedup_noc[n-1] = round((single_latency / max(total)), 2)
    core_latency_noc[n-1] = max(total)
    n = n + 1

print("Grafmeister's core count:",  find_ideal_cores(speedup,0))
print("Grafmeister's NoC core count:",  find_ideal_cores(speedup_noc,1))
print(speedup)
print(speedup_noc)
