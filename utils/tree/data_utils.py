import numpy as np
import re
import networkx as nx


# define parameters
nuc_names = ['A', 'C', 'G', 'T']
transform = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
regex = re.compile('[ACGT]')


def get_char_data(name='infile'):
    """
    Returns numerical representation of dna sequences, number of sequences (N), length of sequences (M).
    A: 0, T: 1, C: 2, G: 3
    """
    regex = re.compile('[ATCG]')
    # transform = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    sequences, N, M = parse_data(name)
    numeric_sequences = np.chararray((N, M), unicode=True)
    n = 0
    for seq in sequences:
        sites = re.findall(regex, seq)
        if len(sites) == 0 or sites is None:
            continue
        sites = sites[-M:]
        numeric_sequences[n, :] = np.array([site for site in sites])
        n += 1
    return numeric_sequences, N, M


def parse_data(name='infile'):
    f = open(name, 'r')
    sequences = []
    meta_data = f.readline().split()
    num_sequences = int(meta_data[0])
    len_sequences = int(meta_data[1])

    for i in range(len_sequences):
        seq = f.readline().strip('\n')
        sequences.append(seq)

    return sequences, num_sequences, len_sequences

def simulate_seq(tree, evo_model, ndata=10):
    """simulate sequences given the tree topology and rate matrices"""

    n_nodes = len(tree)
    root = n_nodes - 1
    n_leaves = (n_nodes + 1) // 2
    pt_matrix = [np.zeros((4, 4)) for i in range(2 * n_leaves - 2)]

    # do postorder tree traversal to compute the transition matrices
    for node in nx.dfs_postorder_nodes(tree, root):
        if not tree.nodes[node]['type'] == 'root':
            t = tree.nodes[node]['t']
            pt_matrix[node] = evo_model.trans_matrix(t)

    simuData = []
    status = [''] * (2 * n_leaves - 1)
    for run in range(ndata):
        for node in nx.dfs_preorder_nodes(tree, root):
            if tree.nodes[node]['type'] == 'root':
                status[node] = np.random.choice(4, size=1, p=evo_model.stat_prob)[0]
            else:
                parent = tree.nodes[node]['parent']
                status[node] = np.random.choice(4, size=1, p=pt_matrix[node][status[parent]])[0]

        simuData.append([nuc_names[i] for i in status[:n_leaves]])

    return np.transpose(simuData)


def simulate_seq_all(tree, evo_model, ndata=10):
    """simulate sequences given the tree topology and rate matrices"""

    n_nodes = len(tree)
    root = n_nodes - 1
    n_leaves = (n_nodes + 1) // 2
    pt_matrix = [np.zeros((4, 4)) for i in range(2 * n_leaves - 2)]

    # do postorder tree traversal to compute the transition matrices
    for node in nx.dfs_postorder_nodes(tree, root):
        if not tree.nodes[node]['type'] == 'root':
            t = tree.nodes[node]['t']
            pt_matrix[node] = evo_model.trans_matrix(t)

    simuData = []
    status = [''] * (2 * n_leaves - 1)
    for run in range(ndata):
        for node in nx.dfs_preorder_nodes(tree, root):
            if tree.nodes[node]['type'] == 'root':
                status[node] = np.random.choice(4, size=1, p=evo_model.stat_prob)[0]
            else:
                parent = tree.nodes[node]['parent']
                status[node] = np.random.choice(4, size=1, p=pt_matrix[node][status[parent]])[0]

        simuData.append([nuc_names[i] for i in status[:ndata]])

    return np.transpose(simuData)

# def simulate_seq(tree, evo_model, ndata=10):
#     n_nodes = len(tree)
#     rate=np.zeros((n_nodes, n_nodes,ndata))
#     #compare with random, gamma, and perhaps one more thing?
#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             for z in range(ndata):
#                 temp_rate=np.random.gamma(3, 10, ndata)
#                 max=np.max(temp_rate)
#                 rate[i][j]=temp_rate/max
#     root = n_nodes - 1
#     n_leaves = (n_nodes + 1) // 2
#     pt_matrix = [np.zeros((4, 4)) for i in range(2 * n_leaves - 2)]
    

#     # do postorder tree traversal to compute the transition matrices
#     for node in nx.dfs_postorder_nodes(tree, root):
#         if not tree.nodes[node]['type'] == 'root':
#             t = tree.nodes[node]['t']
#             pt_matrix[node] = evo_model.trans_matrix(t)

#     simuData = []
#     status = [''] * (2 * n_leaves - 1)

#     for run in range(ndata):
#         iter=0
        
#         for node in nx.dfs_preorder_nodes(tree, root):
#             iter=iter+1
#             if tree.nodes[node]['type'] == 'root':
#                 status[node] = np.random.choice(4, size=1, p=evo_model.stat_prob)[0]
                
#             else:
#                 parent = tree.nodes[node]['parent']
#                 slow=rate[parent][node][run]
#                 fast=1-slow
#                 expect=0.1*slow+10*fast
#                 # print("old:",pt_matrix[node][status[parent]])
#                 # print("rate", expect)
#                 prob=expect*pt_matrix[node][status[parent]]
#                 # print("prob:",prob)
                
#                 # if run>0 and node <n_leaves:
#                 #     prev=simuData[run-1][node]
#                 #     previous=transform[prev]
                
#                 prev=np.argmax(prob)    
#                 prob[prev]=pt_matrix[node][status[parent]][prev]
                
#                 # print("up", prob)
#                 new_p=prob/(np.sum(prob, axis=0))
#                 # print("new_", new_p)
                
#                 status[node] = np.random.choice(4, size=1, p=new_p)[0]#here is where we can combine the rate
         
#                 prev=status[node]
    
#         simuData.append([nuc_names[i] for i in status[:n_leaves]])
    
#     data_Return=np.transpose(simuData)
#     # temp=np.full(fill_value=0.5, dtype=float, shape=(len(tree)-1))
#     n_sites = len(simuData)
#     prob_low = np.zeros((n_nodes, n_nodes,n_sites))
#     n_sites = len(simuData)
    
#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             for z in range(n_sites):
#                 prob_low[i][j][z]=0.1
   

#     return data_Return, prob_low



# Test: simulate sequences for leaf nodes and interior nodes
# def simulate_seq(tree, evo_model, ndata=10):
#     n_nodes = len(tree)
#     root = n_nodes - 1
#     n_leaves = (n_nodes + 1) // 2

#     # Initialize the rate matrix
#     rate = np.zeros((n_nodes, n_nodes, ndata))
#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             temp_rate = np.random.gamma(3, 10, ndata)
#             max_val = np.max(temp_rate)
#             rate[i, j] = temp_rate / max_val

#     # Initialize transition matrices for each node
#     pt_matrix = [np.zeros((4, 4)) for _ in range(n_nodes)]

#     # Step 1: Postorder Traversal to Compute Transition Matrices for Interior Nodes
#     for node in nx.dfs_postorder_nodes(tree, root):
#         if not tree.nodes[node].get('type') == 'root':
#             t = tree.nodes[node]['t']  # Evolutionary time associated with node
#             pt_matrix[node] = evo_model.trans_matrix(t)

#     # Step 2: Preorder Traversal to Simulate Sequences for All Nodes
#     nuc_names = ['A', 'C', 'G', 'T']
#     status = [''] * n_nodes
#     all_sequences = []

#     for run in range(ndata):
#         # Initialize an empty list to store sequences for each run
#         run_sequences = [''] * n_nodes

#         for node in nx.dfs_preorder_nodes(tree, root):
#             if tree.nodes[node].get('type') == 'root':
#                 # Initialize root node sequence using stationary probabilities
#                 status[node] = np.random.choice(4, size=1, p=evo_model.stat_prob)[0]

#             else:
#                 # Generate sequence for non-root nodes
#                 parent = tree.nodes[node]['parent']
#                 slow = rate[parent][node][run]
#                 fast = 1 - slow
#                 expect = 0.1 * slow + 10 * fast

#                 # Apply transition matrix and compute probability
#                 prob = expect * pt_matrix[node][status[parent]]

#                 # Adjust the probabilities to make sure they sum to 1
#                 prev = np.argmax(prob)
#                 prob[prev] = pt_matrix[node][status[parent]][prev]
#                 new_p = prob / (np.sum(prob, axis=0))

#                 # Sample a nucleotide based on computed probabilities
#                 status[node] = np.random.choice(4, size=1, p=new_p)[0]

#             # Convert the status to nucleotide and store it for the current run
#             run_sequences[node] = nuc_names[status[node]]

#         # Append sequences for the current run to all sequences
#         all_sequences.append(run_sequences)

#     # Convert the list of all sequences to a numpy array
#     all_sequences_np = np.array(all_sequences).T  # Transpose to get each row as a node's sequence

#     return all_sequences_np