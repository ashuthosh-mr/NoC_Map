import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python graphmeister.py <number_of_nodes> <graph>")
    sys.exit(1)

n = int(sys.argv[1])
graph = sys.argv[2]

import importlib
module_name = 'data_' + graph
importlib.invalidate_caches()  # Optional: Clear import caches if needed
data_module = importlib.import_module(module_name)
h_graph_edges = data_module.h_graph_edges
h_graph_nodes = data_module.h_graph_nodes

#vertices and edges retrieval
V=len(h_graph_nodes)-1
E=len(h_graph_edges)

#make hown
hown = [0]*V
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
#debug
#print(vcount)
#print(ecount)
#print(crosscom)
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
print(sum(total)/len(total))
