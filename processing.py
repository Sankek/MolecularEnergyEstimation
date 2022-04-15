import numpy as np

import torch
import torch_geometric.data as tgd

from sklearn.preprocessing import LabelBinarizer

def one_hot_encode(y, unique):
    binarizer = LabelBinarizer().fit(unique)
    one_hot_encoded = binarizer.transform(y)
    
    return one_hot_encoded    

def get_edge_index(positions, link_cutoff=10):
    link_cutoff = 10
    distances = np.linalg.norm((np.expand_dims(positions, 1) - np.expand_dims(positions, 0)), axis=2)
    mask = distances < link_cutoff
    np.fill_diagonal(mask, False)
    
    import scipy
    sparse_matrix = scipy.sparse.csr_matrix(mask.astype(int))
    edge_index, edge_attrs = tg.utils.from_scipy_sparse_matrix(sparse_matrix)
    
    return edge_index

def get_data(row, unique_symbols):
    n_atoms = row.natoms  
    numbers = row.numbers 
    positions = row.positions
    energy_array = row.data.get('energy')
    energy = torch.from_numpy(energy_array.astype(np.float32)) if energy_array else None

    symbols = row.symbols
    one_hot_symbols = one_hot_encode(symbols, unique_symbols)


    # x = torch.from_numpy(
    #     np.concatenate(
    #         (positions.astype(np.float32), one_hot_symbols), 
    #         dtype=np.float32, axis=1
    #     )
    # )

    # data = tgd.Data(x=x, edge_index=get_edge_index(positions), num_nodes=n_atoms, y=energy)
    
    data = tgd.Data(z=torch.from_numpy(numbers.astype(np.int64)), pos=torch.from_numpy(positions.astype(np.float32)), edge_index=get_edge_index(positions), num_nodes=n_atoms, y=energy)   

    return data