import sys
import numpy as np
from data import h_graph_edges,h_graph_nodes



# Check if the number of command line arguments is valid
if len(sys.argv) != 4:
    print("Usage: python communication.py <number_of_nodes> <graph> <number_of_vertices>")
    sys.exit(1)

# Parse command line arguments
n = int(sys.argv[1])
graph = sys.argv[2]
V = int(sys.argv[3])

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
            receive_total[j]=receive_total[j]+send_local[i][j]
        if(i==j):
            intra_communication[i]=intra_communication[i]+send_local[i][i]
for x, y in zip(send_total, receive_total):
    inter_communication.append(x + y)
for x, y in zip(inter_communication, intra_communication):
    cfactor.append(x / y)

for i in range (0,n-1):
    temp2=temp2+send_local[15][i]
print(temp2)
