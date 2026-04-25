import numpy as np
import torch
import torch as th
import dgl
import os
import cupy as cp
import time
# import tqdm
from tqdm import tqdm
_computed_array = None
_part_array = None
_spmm_method = 0  # 0: no reorder, 1: reorderd, 2: no reorder with samplin
_sampling_method = 1
part_id = None


csr_sort_kernel = cp.RawKernel(r'''
extern "C" __global__
void csr_sort_by_deg(const int* indptr, int* indices,
                     const int* deg, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    int len = end - start;

    // Simple selection sort on GPU per row
    for (int i = 0; i < len - 1; ++i) {
        int max_idx = i;
        int max_deg = deg[indices[start + i]];
        for (int j = i + 1; j < len; ++j) {
            int cur_deg = deg[indices[start + j]];
            if (cur_deg > max_deg) {
                max_deg = cur_deg;
                max_idx = j;
            }
        }
        // Swap
        if (max_idx != i) {
            int temp = indices[start + i];
            indices[start + i] = indices[start + max_idx];
            indices[start + max_idx] = temp;
        }
    }
}
''', 'csr_sort_by_deg')

def gpu_sort_csr_by_deg(indptr, indices, deg):
    indptr = cp.asarray(indptr, dtype=cp.int32)
    indices = cp.asarray(indices, dtype=cp.int32)
    deg = cp.asarray(deg, dtype=cp.int32)

    num_rows = indptr.size - 1

    threads_per_block = 128
    blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block

    csr_sort_kernel((blocks_per_grid,), (threads_per_block,),
                    (indptr, indices, deg, num_rows))

    return indices



# def sort_neighbors_by_deg_then_part_csr(indptr, indices, deg, part):
def sort_neighbors_segmented_csr(indptr, indices, deg, part):
  indptr = cp.asarray(indptr)
  indices = cp.asarray(indices)
  deg = cp.asarray(deg)
  part = cp.asarray(part)

  sorted_indices = cp.empty_like(indices)

  for i in tqdm(range(indptr.size - 1), desc="Sorting neighbors"): 
      start = indptr[i].item()
      end = indptr[i + 1].item()

      if start == end:
          continue

      nbrs = indices[start:end]
      degs = deg[nbrs]
      parts = part[nbrs]

      # Step 1: stable sort by degree DESC
      sort_by_deg = cp.argsort(-degs, kind='stable')
      nbrs_deg_sorted = nbrs[sort_by_deg]
      parts_deg_sorted = parts[sort_by_deg]

      # Step 2: stable sort by part ASC
      sort_by_part = cp.argsort(parts_deg_sorted, kind='stable')
      final_sorted = nbrs_deg_sorted[sort_by_part]

      sorted_indices[start:end] = final_sorted

  return sorted_indices


def sort_neighbors_segmented_csr_deg(indptr, indices, deg, part):
  # Ensure everything is on GPU
  indptr = cp.asarray(indptr)
  indices = cp.asarray(indices)
  deg = cp.asarray(deg)
  part = cp.asarray(part)

  sorted_indices = cp.empty_like(indices)

  for i in tqdm(range(indptr.size - 1), desc="Sorting neighbors"):
      start = indptr[i].item()
      end = indptr[i + 1].item()

      if start == end:
          continue  # skip empty rows

      nbrs = indices[start:end]
      parts = part[nbrs]
      degs = deg[nbrs]

      # Stack keys for lexsort: shape must be (2, N)
      # keys = cp.stack([-degs, parts], axis=0)
      # print(f"keys of vertex {i}:", keys)
      # sort_order = cp.lexsort(keys)
      sort_order = cp.argsort(-degs)
      # sort_order = cp.argsort(parts)
      # sort_order = cp.argsort(sort_order)  # descending degree and ascending partition

      sorted_indices[start:end] = nbrs[sort_order]

  return sorted_indices



def sort_neighbors_by_part_and_degree(indptr, indices, deg, part):
    sorted_indices = np.empty_like(indices)
    for i in tqdm(range(len(indptr) - 1), desc="Sorting neighbors"):
        start, end = indptr[i], indptr[i + 1]
        nbrs = indices[start:end]
        # Composite key: (partition ascending, degree descending)
        sort_keys = [(part[n], -deg[n]) for n in nbrs]
        sorted_nbrs = [x for _, x in sorted(zip(sort_keys, nbrs))]
        sorted_indices[start:end] = sorted_nbrs
    return sorted_indices




def metis_partition(G, parts=None, method=None, spmm_reorderd=0, sampling=0, dataset=None, path=None):
    global _computed_array
    global _part_array
    global _spmm_method
    global _sampling_method
    global part_id
    if _computed_array is None and sampling == 0:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        print(G)
        print(type(G))
        print("start")
        Nodes = G.num_nodes() 
        # num_parts = int(Nodes/1024)
        # num_parts = int(Nodes/1024)
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)
        # part = str(parts)
        # print(g)
        _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
        if method is None:
                ###---------------------------start the partitioning------------------------------------------------
                # if os.path.exists(dataset + "_metis_part_" + part + ".npy"):
                #     _computed_array = np.load(dataset + "_metis_part_" + part + ".npy")
                #     print(f"Loaded array from {dataset}_metis_part_{part}.npy")
                #     # torch_tensor = torch.from_numpy(_computed_array)
                #     # _computed_array = torch_tensor
                #     # max_len = max(len(_computed_array), len(b))
                #     # b = np.pad(b, (0, max_len - len(b)), constant_values=0)
                #     # _computed_array = torch.from_numpy(np.stack((_computed_array, b), axis=0))
                #     _computed_array = _computed_array.to(device)
                ###------------------------------end the partitioning------------------------------------------------

                ##------------------------------start the sorting------------------------------------------------
                if os.path.exists(dataset + "_sorte.pth"):
                 # _computed_array = np.load(dataset + "_metis_part_" + part + "_sorted.pth")
                  _computed_array = torch.load(dataset + "_sorted.pth")
                  _computed_array = _computed_array.to(device)
                  print(f"Loaded array from {dataset}_metis_part_{part}_sorted.pth")
                ##----------------------------------end the sorting------------------------------------------------

                else:
                    ###---------------------------start the partitioning------------------------------------------------
                    # _computed_array = dgl.metis_partition_assignment(G, num_parts, balance_ntypes=None, balance_edges=True, mode='k-way', objtype='cut')
                    # # _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=True, mode='k-way', objtype='cut')
                    # np.save(dataset + "_metis_part_" + part + ".npy", _computed_array)
                    # print(f"Created new partion and saved to {dataset}_metis_part_{part}.npy")
                   # # b = np.array([4, 5, 6])
                    # # _computed_array = torch.from_numpy(np.stack((_computed_array, b), axis=0))
                    # _computed_array = _computed_array.to(device)
                    # # torch_tensor = torch.from_numpy(_computed_array)
                     print(f"********** this is else part **************")
                    # ### _computed_array = dgl.ndarray.array(_computed_array)
                    ###------------------------------end the partitioning------------------------------------------------

                    ##---------------------------strat the sorting-------------------------------------------------------
                     start_prep_time = time.time()
                     deg = G.in_degrees().numpy()  # Get in-degrees of nodes
                     # part_id = dgl.metis_partition_assignment(G, parts)
                     # part_id = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=True, mode='k-way', objtype='cut')
                     # part_id = dgl.metis_partition_assignment(G, parts, balance_ntypes=None,  mode='k-way', objtype='vol')
                     indptr = np.array(G.adj_tensors('csr')[0]) # Get the indptr of the adjacency matrix
                     indices = np.array(G.adj_tensors('csr')[1])  # Get the indices of the adjacency matrix
                     #reading weights from file
                     # gsum = np.loadtxt("{path}{dataset}_output.csr_g_sum_output.txt", dtype=np.uint64)
                     gsum = np.loadtxt(f"{path}{dataset}_output.csr_g_sum_output.txt", dtype=np.uint64)
                     print(gsum)
                     tri_weight = gsum[indices]
                     print(tri_weight)
                     # Apply sorting
                     start_sort_time = time.time()
                     # sorted_indices_cp = sort_neighbors_segmented_csr(indptr, indices, deg, part_id)
                     # sorted_indices_cp = sort_neighbors_segmented_csr_deg(indptr, indices, deg, part_id)
                     # sorted_indices_cp = gpu_sort_csr_by_deg(indptr, indices, deg)
                     col_sorted = np.empty_like(indices)

                     for u in range(len(indptr) - 1):
                         start, end = indptr[u], indptr[u+1]

                         # get local slice
                         neigh = indices[start:end]
                         w = tri_weight[start:end]

                         # sort by descending weight
                         order = np.argsort(-w)

                         col_sorted[start:end] = neigh[order]

                     print(col_sorted)
                     end_sort_time = time.time()
                     print(f"Sorting took {end_sort_time - start_sort_time:.2f} seconds")

                     # If you need it on CPU
                     sorted_indices = cp.asnumpy(col_sorted)
                     print("sorted Indices GPU: ", sorted_indices)

                     # sorted_indices = sort_neighbors_by_part_and_degree(indptr, indices, deg, part_id)

                     # print("Sorted Indices:", sorted_indices)
                     _computed_array = sorted_indices
                     print(type(_computed_array))
                     # torch_tensor = torch.to_tensor(_computed_array)  # Convert to torch tensor
                     # torch_tensor = torch.from_numpy(_computed_array)
                     _computed_array = torch.tensor(_computed_array).to('cuda')
                     print(type(_computed_array))
                     # torch_tensor = _computed_array.to_tensor()
                     # b = np.array([4, 5, 6])
                     # _computed_array = torch.from_numpy(np.concatenate((_computed_array, part_id), axis=0))
                     # torch_tensor = torch.from_numpy(_computed_array)
                     # print("type of torch_tensor: ", type(torch_tensor))
                     # print(type(_computed_array))
                     # np.save(dataset + "_metis_part_" + part + "_sorted.npy", _computed_array)
                     # import torch
                     torch.save(_computed_array, dataset + "_sorted.pth")
                     # torch.save(torch_tensor, dataset + "_metis_part_" + part + "_sorted.pth")
                     end_prep_time = time.time()
                     # print(f"Created new partion and saved to {dataset}_metis_part_{part}_sorted.pth")
                     print(f"Preparation took {end_prep_time - start_prep_time:.2f} seconds")
                    ##---------------------------end the sorting------------------------------------------------

                    # _computed_array = torch_tensor
                    # _computed_array = _computed_array.to(device)

        elif method == "rm":
            _computed_array = np.random.randint(0, parts, size=Nodes)
        elif method == "contig":
            l = Nodes // parts   # calculate the number of repeated values for each number
            _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
            for i in range(parts):
                _computed_array[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
        elif method == "metis":
            _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        # _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        print("_computed array: ", _computed_array)
        # print("_computed array size: ", _computed_array.shape)
        #-------------spmm methos is reorderd then it executed-----------------------------------
        if spmm_reorderd == 1:
            _spmm_method = 1
            # num_parts = int(Nodes/1024)
            # _part_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
            # _part_array = _computed_array
            # _part_array = list(map(int, _part_array))
            # _part_array = [[d, i] for i, d in enumerate(_part_array, 0)]
            # _part_array.sort()
            # _part_array = [item[1] for item in _part_array]
            _part_array = th.ones(5)  # create an array of size n filled with ones

            # _part_array = th.tensor(_part_array)
            _part_array = _part_array.to(device)
            print("_part_array from metis sampling",_part_array)
        elif spmm_reorderd == 2:
            _spmm_method = 2
        # context = dgl.cuda.get_context(0)
        # context = dgl.cuda.context(0)
        # _computed_array = np.random.rand(10)
        # _computed_array = np.random.randint(10000, 90001, size=10)
        # _computed_array = _computed_array.astype(np.int64)
        # _computed_array = torch.from_numpy(_computed_array)
        # _computed_array = _computed_array.to(device)
        # _computed_array = dgl.ndarray.array(_computed_array)
        # print("Computed array_nd: ", _computed_array)
        # _computed_array = _computed_array.to(device)
        # Convert NumPy array to DGL tensor
        # _computed_array = dgl.tensor(_computed_array)
        # device = "cuda" if dgl.cuda.is_available() else "cpu"
        # _computed_array = _computed_array.to(device)
        # _computed_array = _computed_array.tolist
        _sampling_method = sampling
        # print("Array computation done and passed to neighbour.py line 631")
    else:
        Nodes = G.num_nodes()
        _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
        # print("_computed array initilize with zeros: ", _computed_array)
        _computed_array = dgl.ndarray.array(_computed_array)
        if spmm_reorderd == 2:
            _spmm_method = 2
        # elif spmm_reorderd == 0:
            # _spmm_method = 0
    # print("spmm_methods: ",_spmm_method)
    return _computed_array, _sampling_method

def return_array():
    global _part_array
    global _spmm_method
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
    # if _part_array is None:
        # print("Array not initilized")
    # print("spmm: ",_spmm_method)
    return _part_array, _spmm_method

def return_part_array():
    global _computed_array
    global _sampling_method
    return _computed_array, _sampling_method

def get_part_array(G, parts=None, method=None, spmm_reorderd=0, sampling=0, dataset=None, path=None):
    if _computed_array is None:
        return metis_partition(G, parts, method, spmm_reorderd, sampling, dataset, path)
    else:
        return return_part_array()

def spmm_part_array():
    return return_array()

def get_part_id():
    global part_id
    if part_id is None:
        return None
    else:
        return part_id
