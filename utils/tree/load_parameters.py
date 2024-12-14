from utils.tree.data_utils import simulate_seq, simulate_seq_all
from utils.tree.tree_utils import create_tree
from utils.tree.substitution_models import JukesCantor
from utils.tree import sem
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix


bases ={'A': 0, 'C': 1, 'G': 2, 'T':3 }


def one_hot(sequences):
    # print(f'sequences: {sequences}')
    # print(f'type: {type(sequences)}')
    bases ={'A': 0, 'C': 1, 'G': 2, 'T':3 }
    max_length = max(len(seq) for seq in sequences)
    
    # Initialize the one-hot encoded array with zeros
    encoded = np.zeros((len(sequences), max_length, 4), dtype=int)
    
    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq):
            # print(f'nucleotide: {nucleotide}')
            # print(f'type: {type(nucleotide)}')
            if nucleotide in bases:
                encoded[i, j, bases[nucleotide]] = 1
    
    return encoded

def compute_log_likelihood(seq1, seq2, model):
    """
    Compute log-likelihood of observing seq2 given seq1 under the Jukes-Cantor model.
    
    Parameters:
        seq1 (list[int]): Encoded reference sequence (0 for A, 1 for C, 2 for G, 3 for T).
        seq2 (list[int]): Encoded observed sequence (0 for A, 1 for C, 2 for G, 3 for T).
        model (JukesCantor): Instance of Jukes-Cantor model.
    
    Returns:
        float: Log-likelihood of the sequences.
    """
    # Ensure sequences are of equal length
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length.")
    # print(seq1)

    # Count substitutions (S_ij matrix)
    S_ij = np.zeros((4, 4), dtype=int)
    for i, j in zip(seq1, seq2):
        S_ij[i, j] += 1

    # Optimize branch length
    t_opt = model.optimize_t(S_ij)
    P_t = model.trans_matrix(t_opt)

    # Compute log-likelihood
    log_likelihood = 0
    for i, j in zip(seq1, seq2):
        log_likelihood += np.log(P_t[i, j])

    return np.sum(log_likelihood)


def build_weighted_adjacency_matrix(one_hot_sequences, model):
    """
    Build a weighted adjacency matrix from one-hot encoded sequences.
    
    Parameters:
        one_hot_sequences (list of numpy.ndarray): List of one-hot encoded sequences.
    
    Returns:
        numpy.ndarray: Weighted adjacency matrix (shape: N x N).
    """
    num_sequences = len(one_hot_sequences)
    adjacency_matrix = np.zeros((num_sequences, num_sequences))
    
    for i in range(num_sequences):
        for j in range(num_sequences):
            if i != j:
                log_likelihood = compute_log_likelihood(
                    one_hot_sequences[i], one_hot_sequences[j], model
                )
                # print(log_likelihood)
                adjacency_matrix[i, j] = log_likelihood
            else:
                adjacency_matrix[i, j] = 0  # No self-loops (optional)
    
    return np.exp(adjacency_matrix)

def generate_node_features(n_leaves, alpha=0.1):
    # tree, opt = create_tree(n_leaves, scale=0.1)
    tree,opt = sem.randomt_tree(n_leaves=n_leaves, scale=0.1)
    evo_model = JukesCantor(alpha=alpha)
    sim_seq = simulate_seq(tree, evo_model, len(tree))
    encoded_seq = one_hot(sim_seq)
    
    features_reduced = np.argmax(encoded_seq, axis=-1)
    return features_reduced, opt


def generate_weighted_adj_matrix(n_leaves, alpha = 0.1):
    """
    Generate weight adjancy matrix and one-hot encoded sequences
    

    Returns:
        adj_matrix: numpy.array
        encoded_seq: numpy.array
    """
    encoded_seq, opt = generate_node_features(n_leaves=n_leaves, alpha=alpha)

    model = JukesCantor(alpha)
    
    weighted_adj_matrix = build_weighted_adjacency_matrix(encoded_seq, model)

    return weighted_adj_matrix, opt


def get_mst(weighted_adj_matrix, t_opts):
    # W is a symmetric (2N - 1)x(2N - 1) matrix with MI entries. Last entry is link connection to root.
    G = nx.Graph()
    W = weighted_adj_matrix

    n_nodes = W.shape[0]
    n_nodes -= 1
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (W[i, j] == -np.inf):
                continue

            t = t_opts[i, j]  # nx.shortest_path_length(tree, i, j, weight='t')
            G.add_edge(i, j, weight=W[i, j], t = t)
    
    mst = nx.maximum_spanning_tree(G)
  
    return mst


def generate_unweighted_adj_matrix(mst, num_nodes):
    edges = np.array(list(mst.edges)).T 
    print(edges.shape)
    # initialize an n_nodes x n_nodes adjacency matrix with zeros
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    print("Adjacency matrix shape:", adj_matrix.shape)
    
    # populate the adjacency matrix based on edge_index
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1  

    return adj_matrix


# def generate_data(n_leaves=20, alpha=0.1):

#     node_features, t_opts = generate_node_features(n_leaves=n_leaves, alpha=alpha)
#     weighted_adj_matrix, t_opts = generate_weighted_adj_matrix(n_leaves=n_leaves, alpha=alpha)
#     mst = get_mst(weighted_adj_matrix, t_opts)
#     adj_matrix = generate_unweighted_adj_matrix(mst, num_nodes=node_features.shape[0])

#     if node_features.shape[0] != adj_matrix.shape[0]:
#         print("Inconsistant shape between node_features and adjancy matrix.")

#     if adj_matrix.shape[0] != adj_matrix.shape[1]:
#         print("Invalid adjancy matrix.")

#     return sp.csr_matrix(adj_matrix), node_features

def get_labels(tree):
    #labels for node (root:0, leaf:1, internal:2)
    type_to_label = {
    'root': 0,
    'leaf': 1,
    'internal': 2}

    labels = []
    for node in tree.nodes:
        node_type = tree.nodes[node]['type']
        labels.append(type_to_label[node_type])
    labels = np.array(labels)

    return labels

def generate_data(n_leaves, n_features,  alpha=0.1):
    # generate a tree
    tree, opt = sem.randomt_tree(n_leaves=n_leaves, scale=0.1)
    # generate node features
    evo_model = JukesCantor(alpha=alpha)
    sim_seq = simulate_seq_all(tree, evo_model, n_features)
    print(f'sim_seq shape: {sim_seq.shape}')

    encoded_seq = one_hot(sim_seq)
    features_reduced = np.argmax(encoded_seq, axis=-1)
    print(f'features shape: {features_reduced.shape}')

    #adjancy matrix
    adj_matrix_sparse = nx.adjacency_matrix(tree)
    adj_matrix = adj_matrix_sparse.toarray()
    print(f'adj matrix shape: {adj_matrix.shape}')
    
    labels = get_labels(tree)

    return sp.csr_matrix(adj_matrix), features_reduced, labels, opt