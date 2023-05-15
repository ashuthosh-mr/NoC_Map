import sys

def calculate_hops(node1, node2, network_size):
    hops_forward = (node2 - node1) % network_size
    hops_backward = (node1 - node2) % network_size
    return min(hops_forward, hops_backward)

# Check if the number of command line arguments is valid
if len(sys.argv) != 3:
    print("Usage: python torus_hops.py <number_of_nodes> <temporary_argument>")
    sys.exit(1)

# Parse command line arguments
network_size = int(sys.argv[1])
temporary_argument = sys.argv[2]

# Calculate number of hops for all node pairs
for node1 in range(1,network_size+1):
    for node2 in range(1,network_size+1):
        num_hops = calculate_hops(node1, node2, network_size)
        print(f"Number of hops between {node1} and {node2} is: {num_hops}")
