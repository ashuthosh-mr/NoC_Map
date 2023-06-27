import sys
import numpy as np
import matplotlib.pyplot as plt

#function definitions

#binary tree
# Check if the number of command line arguments is valid
if len(sys.argv) != 2:
    print("Usage: python bfsrodinia.py <graph>")
    sys.exit(1)

# Parse command line arguments
graph = sys.argv[1]
import importlib
module_name = 'data_' + graph
importlib.invalidate_caches()  # Optional: Clear import caches if needed
data_module = importlib.import_module(module_name)
h_graph_edges = data_module.h_graph_edges
h_graph_nodes = data_module.h_graph_nodes

#bfs
V=3031
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
vcount = [0]*4
ecount = [0]*4
result = 0
E = len(h_graph_edges)
m = 0
for i in range(0,V):
    result = result + h_graph_nodes[i+1]-h_graph_nodes[i]
    if(result>=3237):
        ecount[m]=result
        result=0
        m = m + 1
    else:
        vcount[m]=vcount[m]+1
ecount[m]=result
print(vcount)
print(ecount)
print(sum(ecount))
print(h_graph_nodes[0])

print('int h_own1[3031]={')
for i in range(0,V):
    if(i%100==0):
        print('')
    if (i<795):
        print(i,',',end='')
    elif(795<=i<795+834):
        print('-2,',end='')
    elif(795+834<=i<795+834+832):
        print('-3,',end='')
    else:
        print('-4,',end='')
print('};')
print('')
print('int h_graph_nodes1[796]={0,',end='')
result=0
for i in range(0,795):
    if(i%100==0):
        print('')
    result = result + h_graph_nodes[i+1] - h_graph_nodes[i]
    print(result,',',end='')
print('};')
print('')
print('int h_graph_edges1[3237]={',end='')
result=0
nexit=0
for i in range(0,795):
    for j in range(h_graph_nodes[i],h_graph_nodes[i+1]):
        result = h_graph_edges[j]
        nexit=nexit+1
        if(nexit%100==0):
            print('')
        print(result,',',end='')
print('};')

#second
p=0
print('int h_own2[3031]={')
for i in range(0,V):
    if(i%100==0):
        print('')
    if (i<795):
        print('-1,',end='')
    elif(795<=i<795+834):
        print(p,',',end='')
        p=p+1
    elif(795+834<=i<795+834+832):
        print('-3,',end='')
    else:
        print('-4,',end='')
print('};')
print('')
print('int h_graph_nodes2[835]={0,',end='')
result=0
for i in range(795,795+834):
    if(i%100==0):
        print('')
    result = result + h_graph_nodes[i+1] - h_graph_nodes[i]
    print(result,',',end='')
print('};')
print('')
print('int h_graph_edges2[3238]={',end='')
result=0
nexit=0
for i in range(795,795+834):
    for j in range(h_graph_nodes[i],h_graph_nodes[i+1]):
        result = h_graph_edges[j]
        nexit=nexit+1
        if(nexit%100==0):
            print('')
        print(result,',',end='')
print('};')

#third
p=0
print('int h_own3[3031]={')
for i in range(0,V):
    if(i%100==0):
        print('')
    if (i<795):
        print('-1,',end='')
    elif(795<=i<795+834):
        print('-2,',end='')
    elif(795+834<=i<795+834+832):
        print(p,',',end='')
        p=p+1
    else:
        print('-4,',end='')
print('};')
print('')
print('int h_graph_nodes3[833]={0,',end='')
result=0
for i in range(795+834,795+834+832):
    if(i%100==0):
        print('')
    result = result + h_graph_nodes[i+1] - h_graph_nodes[i]
    print(result,',',end='')
print('};')
print('')
print('int h_graph_edges3[3238]={',end='')
result=0
nexit=0
for i in range(795+834,795+834+832):
    for j in range(h_graph_nodes[i],h_graph_nodes[i+1]):
        result = h_graph_edges[j]
        nexit=nexit+1
        if(nexit%100==0):
            print('')
        print(result,',',end='')
print('};')

#fourth
p=0
print('int h_own4[3031]={')
for i in range(0,V):
    if(i%100==0):
        print('')
    if (i<795):
        print('-1,',end='')
    elif(795<=i<795+834):
        print('-2,',end='')
    elif(795+834<=i<795+834+832):
        print('-3,',end='')
    else:
        print(p,',',end='')
        p=p+1
print('};')
print('')
print('int h_graph_nodes4[571]={0,',end='')
result=0
for i in range(795+834+832,V):
    if(i%100==0):
        print('')
    result = result + h_graph_nodes[i+1] - h_graph_nodes[i]
    print(result,',',end='')
print('};')
print('')
print('int h_graph_edges4[3186]={',end='')
result=0
nexit=0
for i in range(795+834+832,V):
    for j in range(h_graph_nodes[i],h_graph_nodes[i+1]):
        result = h_graph_edges[j]
        nexit=nexit+1
        if(nexit%100==0):
            print('')
        print(result,',',end='')
print('};')
