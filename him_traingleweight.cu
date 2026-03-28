#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

// Kernel to sort adjacency lists in-place on the GPU.
// Launches one thread per vertex. Each thread sorts its own list.
__global__ void sort_adj_lists_kernel(const int* g_row_ptr, int* g_col_index, int num_vertices) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex_id < num_vertices) {
        int start_idx = g_row_ptr[vertex_id];
        int end_idx = g_row_ptr[vertex_id + 1];
        int degree = end_idx - start_idx;

        if (degree > 1) {
            // Using a simple insertion sort. This is efficient for the small
            // adjacency lists typically found in many graphs, and it's done in parallel
            // for all vertices.
            for (int i = start_idx + 1; i < end_idx; ++i) {
                int key = g_col_index[i];
                int j = i - 1;
                while (j >= start_idx && g_col_index[j] > key) {
                    g_col_index[j + 1] = g_col_index[j];
                    j = j - 1;
                }
                g_col_index[j + 1] = key;
            }
        }
    }
}

// Device function to perform binary search on a sorted array (a vertex's neighbor list)
__device__ __forceinline__ bool does_edge_exist(const int* neighbors, int num_neighbors, int target_neighbor) {
    int low = 0;
    int high = num_neighbors - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (neighbors[mid] == target_neighbor) {
            return true;
        } else if (neighbors[mid] < target_neighbor) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return false;
}

// Kernel to count triangles for each edge
__global__ void count_triangles_per_edge(
    const int* g_row_ptr, 
    const int* g_col_index_unsorted, // Used to get the correct destination node 'v'
    const int* g_col_index_sorted,   // Used to search neighbor lists efficiently
    const int* g_edge_to_src_map, 
    int* g_edge_weights, 
    int num_edges
) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_id < num_edges) {
        // BUG FIX: Get 'u' and 'v' from the ORIGINAL, UNSORTED graph data
        // to ensure edge_id maps to the correct (u, v) pair.
        int u = g_edge_to_src_map[edge_id];
        int v = g_col_index_unsorted[edge_id];

        // BUG FIX: Get neighbor lists from the new, SORTED graph data
        // to ensure the binary search in does_edge_exist works correctly.
        const int* u_neighbors = &g_col_index_sorted[g_row_ptr[u]];
        int u_degree = g_row_ptr[u + 1] - g_row_ptr[u];

        const int* v_neighbors = &g_col_index_sorted[g_row_ptr[v]];
        int v_degree = g_row_ptr[v + 1] - g_row_ptr[v];
        
        if (u_degree == 0 || v_degree == 0) {
            g_edge_weights[edge_id] = 0;
            return;
        }

        int triangle_count = 0;

        // Iterate through neighbors of u and check if they are also neighbors of v.
        for (int i = 0; i < u_degree; ++i) {
            int w = u_neighbors[i];
            if (does_edge_exist(v_neighbors, v_degree, w)) {
                triangle_count++;
            }
        }
        g_edge_weights[edge_id] = triangle_count;
    }
}

// Kernel to apply the weight formula
__global__ void apply_weight_formula(int* g_edge_weights, int num_edges) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_id < num_edges) {
        int triangle_count = g_edge_weights[edge_id];
        g_edge_weights[edge_id] = 1 + (triangle_count * 10);
    }
}

// Helper function to read a file of integers into a vector
std::vector<int> readIntFile(const std::string& filename) {
    std::vector<int> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    int value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_directory>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];

    // Read Graph Data from Directory
    auto h_row_ptr_vec = readIntFile(dir + "/row.txt");
    auto h_col_index_vec = readIntFile(dir + "/column.txt");
    
    int row_ptr_size = h_row_ptr_vec.size();
    int col_idx_size = h_col_index_vec.size();
    int num_vertices = row_ptr_size - 1;
    int num_edges = col_idx_size;

    if (num_vertices <= 0 || num_edges <= 0) {
        std::cerr << "Error: Graph is empty or invalid." << std::endl;
        return 1;
    }

    // Allocate Host Memory
    int *h_row_ptr = (int *)malloc(row_ptr_size * sizeof(int));
    int *h_col_index = (int *)malloc(col_idx_size * sizeof(int));
    int *h_edge_weights = (int *)malloc(num_edges * sizeof(int));
    int *h_edge_to_src_map = (int *)malloc(num_edges * sizeof(int));

    if (!h_row_ptr || !h_col_index || !h_edge_weights || !h_edge_to_src_map) {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }
    
    memcpy(h_row_ptr, h_row_ptr_vec.data(), row_ptr_size * sizeof(int));
    memcpy(h_col_index, h_col_index_vec.data(), col_idx_size * sizeof(int));

    // Create a map from each edge's index to its source vertex
    int edge_idx = 0;
    for (int i = 0; i < num_vertices; i++) {
        for (int j = h_row_ptr[i]; j < h_row_ptr[i+1]; j++) {
            if(edge_idx < num_edges) {
                h_edge_to_src_map[edge_idx] = i;
                edge_idx++;
            }
        }
    }

    // Allocate GPU Memory
    int *d_row_ptr, *d_col_index_unsorted, *d_col_index_sorted, *d_edge_weights, *d_edge_to_src_map;
    cudaMalloc(&d_row_ptr, row_ptr_size * sizeof(int));
    cudaMalloc(&d_col_index_unsorted, num_edges * sizeof(int));
    cudaMalloc(&d_col_index_sorted, num_edges * sizeof(int));
    cudaMalloc(&d_edge_weights, num_edges * sizeof(int));
    cudaMalloc(&d_edge_to_src_map, num_edges * sizeof(int));

    // Transfer Data from CPU to GPU
    cudaMemcpy(d_row_ptr, h_row_ptr, row_ptr_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index_unsorted, h_col_index, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index_sorted, h_col_index, num_edges * sizeof(int), cudaMemcpyHostToDevice); // Copy to sorted as well
    cudaMemcpy(d_edge_to_src_map, h_edge_to_src_map, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to sort the 'd_col_index_sorted' array
    printf("Launching GPU kernel to sort adjacency lists...\n");
    int sort_threads_per_block = 256;
    int sort_blocks_per_grid = (num_vertices + sort_threads_per_block - 1) / sort_threads_per_block;
    sort_adj_lists_kernel<<<sort_blocks_per_grid, sort_threads_per_block>>>(d_row_ptr, d_col_index_sorted, num_vertices);
    
    cudaDeviceSynchronize(); 
    printf("...Sorting complete.\n");

    // Setup Kernel Launch Configuration
    int threads_per_block = 256;
    int blocks_per_grid = (num_edges + threads_per_block - 1) / threads_per_block;
    
    printf("Processing %d edges to count triangles...\n", num_edges);

    // Launch Kernel to count triangles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    count_triangles_per_edge<<<blocks_per_grid, threads_per_block>>>(d_row_ptr, d_col_index_unsorted, d_col_index_sorted, d_edge_to_src_map, d_edge_weights, num_edges);
    
    apply_weight_formula<<<blocks_per_grid, threads_per_block>>>(d_edge_weights, num_edges);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_edge_weights, d_edge_weights, num_edges * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Kernels finished in %.4f seconds.\n", milliseconds / 1000.0);

    const std::string out_filename = "./TriangleOutput/triangle_weights_csr.txt";
    FILE *outFile = fopen(out_filename.c_str(), "w");
    if (outFile == NULL) {
        printf("Error opening output file!\n");
    } else {
        for (int i = 0; i < num_edges; i++) fprintf(outFile, "%d\n", h_edge_weights[i]);
        fclose(outFile);
        printf("Updated weights saved to %s\n", out_filename.c_str());
    }

    // Free Memory
    free(h_row_ptr);
    free(h_col_index);
    free(h_edge_weights);
    free(h_edge_to_src_map);
    cudaFree(d_row_ptr);
    cudaFree(d_col_index_unsorted);
    cudaFree(d_col_index_sorted);
    cudaFree(d_edge_weights);
    cudaFree(d_edge_to_src_map);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
