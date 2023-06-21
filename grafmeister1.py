import sys
import numpy as np
import matplotlib.pyplot as plt

#function definitions

def arbit_latency(n):
    return ((n-1)*3)+((n-1)*2)+2


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
    
    tcopy = [0]*n
    scopy = [0]*n
    ccopy = [0]*n
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
    ############
    #Traversal latency:
    for i in range(0,n):
        traversal[i]=(vcount[i]*c1)+((ecount[i]-vcount[i])*c2)
    #############Sync latency
    for i in range(0,n):
        temp1=(max(ecount)-min(ecount))/ecount[i]
        temp2=(vcount[i]*c1)+((ecount[i]-vcount[i])*c2)+(arbit_latency(n)*crosscom[i])
        sync[i]=int(temp1*temp2)+((2*iter)*(((n-1)*arbit_latency(n))+arbit_latency(n)+arbit_latency(n)))
    ############communication latency
    #for i in range(0,n):
     #   temp1=(crosscom[i]*(arbit_latency(n)+arbit_latency(n)+c3))
      #  temp2=(n-1)*iter*(arbit_latency(n)+arbit_latency(n)+8)
       # communication[i]=temp1+temp2
    ##################3
    temp3 = 0
    temp4 = 0
    for i in range(0,n):
        k = 0
        for j in range(0,n):
            temp2=0
            if(i!=j):
                temp1 = send_local[i][j]
            if(i!=j):
                for k in range(0,n):
                    if(temp1 >= send_local[i][k]):
                        temp2=temp2+1
                temp3 = ((temp1*((arbit_latency(temp2)*2)+c3)))
                communication[i]=temp3+communication[i]
        temp4=(n-1)*iter*(arbit_latency(n)+arbit_latency(n)+8)
        communication[i]=communication[i]+temp4
    '''          
            
        temp1 = send_local[i][k]
        
        for j in range(0,n):
            if(i!=j):
                if(temp1) >= send_local[i][j]):
                    temp2=temp2+1
        k = k+1
        
        
            temp1 = send_local[i][j]
        temp2 = 0
        for j in range(0,n):
            if(i!=j):
                if(temp1 >= send_local[i][j]):
                    temp2 =temp2+1
        for j in range(0,n):
            if(i!=j):
                temp3 = ((temp1*(arbit_latency(temp2)*2))+c3)       
        temp4=(n-1)*iter*(arbit_latency(n)+arbit_latency(n)+8)
        communication[i]=temp3+temp4
        print(temp3)
        print("\n")
    #tcopy
    '''
    ###########total latency
    for i in range(0,n):
        total[i] = traversal[i] + communication[i] + sync[i]

    speedup[n-1] = round((single_latency / max(total)), 2)
    core_latency[n-1] = max(total)
    i = np.argmax(total)
    #print("core "+ str(n-1) + " is:")
    #print(traversal[i])
    #print(sync[i])
    #print(communication[i])
    n = n + 1

measurements = [0]*n
index = []
inputs = 0
with open('./measurement.in') as f3:
    for line in f3:
        Type = line.split()
        measurements[int(Type[0])-1] = int(Type[1])
        index.append(int(Type[0])-1)
        inputs=inputs+1
print("Accuracy:")
for i in range(0,inputs):
    a = measurements[index[i]]
    b = core_latency[index[i]]
    temp1 = max(a,b)
    temp2 = min(a,b)
    print("core "+str(index[i]+1)+" is:")
    print((temp2/temp1)*100)
#print("Grafmeister's core count:",  find_ideal_cores(speedup,0))
#print("Grafmeister's NoC core count:",  find_ideal_cores(speedup_noc,1))
print(core_latency)
print(speedup)
#print(speedup_noc)
