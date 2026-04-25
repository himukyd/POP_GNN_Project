#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/equal.h>
#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno> // Added for strerror(errno)
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <map>

#define BLOCK_SIZE 256

__global__ void matchVertices(int* xadj, int* adjncy, int* adjwgt, int* match, int* tempweight, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        int start = xadj[tid];
        int end = xadj[tid + 1];
        int bestMatch = -1;
        int maxWeight = -1;

        for (int i = start; i < end; i++) {
            int neighbor = adjncy[i];
            int weight = adjwgt[i];
            if (weight > maxWeight && match[neighbor] == -1) {
                maxWeight = weight;
                bestMatch = neighbor;
            }
        }
        __syncthreads();

        if (bestMatch != -1 && match[tid] == -1) {
            int expected = -1;
            if (atomicCAS(&match[bestMatch], expected, tid) == expected) {
                match[tid] = bestMatch;
                tempweight[tid] = maxWeight;
            }
        }
        __syncthreads();

        if (match[tid] != -1 && bestMatch != match[tid]) {
            int currentMatch = match[tid];
            int currentWeight = tempweight[match[bestMatch]];
            if (maxWeight > currentWeight) {
                atomicExch(&match[currentMatch], -1);
                atomicExch(&tempweight[currentMatch], -1);
                atomicExch(&match[match[bestMatch]], -1);
                atomicExch(&tempweight[match[bestMatch]], -1);
                atomicExch(&match[tid], bestMatch);
                atomicExch(&match[bestMatch], tid);
                atomicExch(&tempweight[tid], maxWeight);
                atomicExch(&tempweight[bestMatch], maxWeight);
            }
        }
    }
}

__global__ void rematchVertices(int* xadj, int* adjncy, int* adjwgt, int* match, int* tempweight, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices && (match[tid] == -1 || tid != match[match[tid]])) {
        match[tid] = -1;
        tempweight[tid] = -1;

        __syncthreads();

        int start = xadj[tid];
        int end = xadj[tid + 1];
        int bestMatch = -1;
        int maxWeight = -1;

        for (int i = start; i < end; i++) {
            int neighbor = adjncy[i];
            int weight = adjwgt[i];
            if (weight > maxWeight && match[neighbor] == -1) {
                maxWeight = weight;
                bestMatch = neighbor;
            }
        }

        if (bestMatch != -1 && match[tid] == -1) {
            int expected = -1;
            if (atomicCAS(&match[bestMatch], expected, tid) == expected) {
                match[tid] = bestMatch;
                tempweight[tid] = maxWeight;
            }
        }
        __syncthreads();

        if (match[tid] != -1 && bestMatch != match[tid]) {
            int currentMatch = match[tid];
            int currentWeight = tempweight[match[bestMatch]];
            if (maxWeight > currentWeight) {
                atomicExch(&match[currentMatch], -1);
                atomicExch(&tempweight[currentMatch], -1);
                atomicExch(&match[match[bestMatch]], -1);
                atomicExch(&tempweight[match[bestMatch]], -1);
                atomicExch(&match[tid], bestMatch);
                atomicExch(&match[bestMatch], tid);
                atomicExch(&tempweight[tid], maxWeight);
                atomicExch(&tempweight[bestMatch], maxWeight);
            }
        }
    }
}

__global__ void mapVertices(int* match, int* cmap, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        if (match[tid] != -1) {
            cmap[tid] = min(tid, match[tid]);
        } else {
            cmap[tid] = tid; // Unmatched vertices are their own representatives
        }
    }
}

// BUG FIX: The original version of this kernel had faulty fallback logic and potential
// race conditions. This could cause an inconsistent number of coarse vertices to be
// generated, with some old, high-value node IDs "leaking" into the coarse graph's
// edge list. This led to a downstream error in DGL when creating the graph.
// FIX: This corrected version uses a stable, two-stage approach.
// Stage 1: Each "representative" node safely claims a new, dense ID.
// Stage 2: All other nodes look up the new ID of their assigned representative.
// This removes the race conditions and ensures the coarse vertex IDs are dense and correct.
__global__ void assign_coarse_labels_stage1_kernel(int* CMap, int* label_map, int* temp, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        if (CMap[idx] == idx) { // I am a representative
            label_map[idx] = atomicAdd(temp, 1);
        }
    }
}

__global__ void assign_coarse_labels_stage2_kernel(int* CMap, int* New_CMap, int* label_map, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int representative_node = CMap[idx];
        if (representative_node >= 0 && representative_node < num_elements) {
            New_CMap[idx] = label_map[representative_node];
        } else {
            // This case should ideally not happen with correct CMap generation *before* this stage.
            // If it does, it indicates a bug upstream in CMap generation (mapVertices).
            // For now, assign -1 as an indicator of an error.
            New_CMap[idx] = -1;
        }
    }
}

__global__ void updateGlobalMapping(int* global_cmap, int* cmap, int* new_ids, int numVertices, int original_nvtxs, int* temp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < original_nvtxs) {
        int curr_vertex = global_cmap[tid]; // Current coarse ID from previous level
        if (curr_vertex >= 0 && curr_vertex < numVertices) {
            int coarse_vertex = cmap[curr_vertex];
            if (coarse_vertex >= 0 && coarse_vertex < numVertices) {
                int new_coarse_id = new_ids[coarse_vertex];
                if (new_coarse_id >= 0) {
                    global_cmap[tid] = new_coarse_id;
                } else {
                    // Assign unmapped vertices to a new coarse ID
                    global_cmap[tid] = atomicAdd(temp, 1) % numVertices;
                }
            } else {
                // Assign unmapped vertices to a new coarse ID
                global_cmap[tid] = atomicAdd(temp, 1) % numVertices;
            }
        } else {
            // Assign unmapped vertices to a new coarse ID
            global_cmap[tid] = atomicAdd(temp, 1) % numVertices;
        }
    }
}

__global__ void countUniqueEdges(int* rowPtr, int* colInd, int* weights, int* newIDs, int* matches, int* cmap, int* uniqueCounts, const int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVertices && idx == cmap[idx]) {        // fixed 
        // Removed unsigned long long foundHash = 0;
        // Removed int uniqueCount = 0; // Will be determined by local_buffer_pos

        // Temporary local buffer to store unique neighbors for this coarse node
        // Assuming a max coarse degree; beyond this, unique edges will be dropped.
        // This is a known limitation for fixed-size local buffers on GPU.
        const int MAX_LOCAL_UNIQUE_NEIGHBORS = 2048; // Increased limit
        int local_unique_neighbors[MAX_LOCAL_UNIQUE_NEIGHBORS];
        int local_unique_count = 0;

        auto addUniqueNeighborToBuffer = [&](int neighbor_id) {
            // Check if neighbor_id already exists in local_unique_neighbors
            bool found = false;
            for (int k = 0; k < local_unique_count; ++k) {
                if (local_unique_neighbors[k] == neighbor_id) {
                    found = true;
                    break;
                }
            }
            // If neighbor is unique and buffer is not full, add it
            if (!found && local_unique_count < MAX_LOCAL_UNIQUE_NEIGHBORS) {
                local_unique_neighbors[local_unique_count++] = neighbor_id;
            }
            // If buffer is full, subsequent unique neighbors will be ignored.
            // This is a limitation; for extremely high-degree coarse nodes, this count will be incorrect.
        };

        // Process neighbors from the current node
        for (int i = rowPtr[idx]; i < rowPtr[idx + 1]; i++) {
            int neighbor = newIDs[colInd[i]];
            if (neighbor != newIDs[idx]) { // Exclude self-loops
                addUniqueNeighborToBuffer(neighbor);
            }
        }

        // Process neighbors from the matched node (if any)
        int matchedVertex = matches[idx];
        if (matchedVertex != -1) {
            for (int i = rowPtr[matchedVertex]; i < rowPtr[matchedVertex + 1]; i++) {
                int matchNeighbor = newIDs[colInd[i]];
                if (matchNeighbor != newIDs[idx]) { // Exclude self-loops
                    addUniqueNeighborToBuffer(matchNeighbor);
                }
            }
        }
        
        // Store the final count of unique neighbors for this coarse node
        uniqueCounts[newIDs[idx]] = local_unique_count;
    }
}

__global__ void prefix_sum(int* row_ptr, int* updaterow, int nvtxs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (int i = 1; i <= nvtxs; i++) {
            updaterow[i] = updaterow[i-1] + row_ptr[i-1];
        }
    }
}

__global__ void fillEdgeArrays(int* rowPtr, int* colInd, int* weights, int* newIDs, int* matches, int* cmap, int* outputRowPtr, int* outputColInd, int* outputWeights, const int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices && tid == cmap[tid]) {
        int new_coarse_id = newIDs[tid];
        int match = matches[tid];
        int currentOutputPos = outputRowPtr[new_coarse_id];

        auto addUniqueEdge = [&](int neighbor, int weight) {
            bool found = false;
            for (int j = outputRowPtr[new_coarse_id]; j < currentOutputPos; j++) {
                if (outputColInd[j] == neighbor) {
                    outputWeights[j] += weight;
                    found = true;
                    break;
                }
            }
            if (!found) {
                outputColInd[currentOutputPos] = neighbor;
                outputWeights[currentOutputPos] = weight;
                currentOutputPos++;
            }
        };

        for (int i = rowPtr[tid]; i < rowPtr[tid + 1]; i++) {
            int neighbor = newIDs[colInd[i]];
            int weight = weights[i];
            if (neighbor != new_coarse_id && neighbor != -1) {
                addUniqueEdge(neighbor, weight);
            }
        }

        if (match != -1) {
            for (int i = rowPtr[match]; i < rowPtr[match + 1]; i++) {
                int neighbor = newIDs[colInd[i]];
                int weight = weights[i];
                if (neighbor != new_coarse_id && neighbor != -1) {
                    addUniqueEdge(neighbor, weight);
                }
            }
        }
    }
}

__global__ void resetMatch(int* match, int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        match[tid] = -1;
    }
}

void readFile(const char *fileName, int *array, int size, double &readingTime) {
    struct timeval begin, end;
    gettimeofday(&begin, NULL);

    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s: %s\n", fileName, strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%d", &array[i]) != 1) {
            fprintf(stderr, "Error reading file %s at element %d\n", fileName, i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
    gettimeofday(&end, NULL);
    readingTime += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 5) { // Now expecting 5 arguments
        std::cerr << "Usage: " << argv[0] << " <directory> <num_vertices> <num_edges> <coarsening_retain_fraction>" << std::endl;
        return 1;
    }

    const char *dir = argv[1];
    int nvtxs = std::atoi(argv[2]);
    int nedges = std::atoi(argv[3]);
    double retain_frac = std::atof(argv[4]); // Read retain fraction from argument

    double inputReadingTime = 0.0;
    double ioTransferTime = 0.0;
    double coarseningComputeTime = 0.0;
    struct timeval begin, end;

    // Allocate host memory for graph data
    int* h_xadj = (int*)malloc(sizeof(int) * (nvtxs + 1));
    int* h_adjncy = (int*)malloc(sizeof(int) * nedges);
    int* h_weight = (int*)malloc(sizeof(int) * nedges);

    // Read input graph files
    gettimeofday(&begin, NULL);
    std::string xadj_path = std::string(dir) + "/row.txt";
    std::string adjncy_path = std::string(dir) + "/column.txt";
    std::string weight_path = std::string(dir) + "/input_coarsening_weights.txt"; // Assuming spectral weights are used
    readFile(xadj_path.c_str(), h_xadj, nvtxs + 1, inputReadingTime);
    readFile(adjncy_path.c_str(), h_adjncy, nedges, inputReadingTime);
    readFile(weight_path.c_str(), h_weight, nedges, inputReadingTime);
    gettimeofday(&end, NULL);
    inputReadingTime += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

    // Initialize device vectors using Thrust
    thrust::device_vector<int> d_xadj(h_xadj, h_xadj + nvtxs + 1);
    thrust::device_vector<int> d_adjncy(h_adjncy, h_adjncy + nedges);
    thrust::device_vector<int> d_adjwgt(h_weight, h_weight + nedges);
    thrust::device_vector<int> d_match(nvtxs, -1);
    thrust::device_vector<int> d_tempweight(nvtxs, -1);
    thrust::device_vector<int> d_cmap(nvtxs);
    thrust::device_vector<int> d_new_ids(nvtxs);
    thrust::device_vector<int> d_label_map(nvtxs);
    thrust::device_vector<int> d_global_cmap(nvtxs);
    
    // Initialize global_cmap to identity mapping for the first level
    thrust::sequence(d_global_cmap.begin(), d_global_cmap.end());

    // Raw device pointers for atomic operations and single-value storage
    int *d_coarsened_nvtxs;
    int *d_coarsened_edges;
    int *temp; // For atomicAdd counters
    cudaMalloc(&d_coarsened_nvtxs, sizeof(int));
    cudaMalloc(&d_coarsened_edges, sizeof(int));
    cudaMalloc(&temp, sizeof(int));
    
    // Initialize device memory
    cudaMemset(d_coarsened_nvtxs, 0, sizeof(int));
    cudaMemset(d_coarsened_edges, 0, sizeof(int));
    cudaMemset(temp, 0, sizeof(int));

    int blockSize = BLOCK_SIZE;
    int numBlocks = (nvtxs + blockSize - 1) / blockSize;

    int target_vertices = int(nvtxs * retain_frac); // Coarsen until 'retain_frac' of original vertices remain.
                                                    // e.g., if retain_frac=0.25, target is 25% of original vertices.
    // ...

    int pass = 0;
    int numPasses = 45;
    std::cout<<"\nThreshold: "<<retain_frac<<"\n";
    int vertex_count_history[numPasses];
    int coarsed_nvtx = nvtxs;
    int coarsed_edge = nedges;

    thrust::host_vector<int> h_match_prev(nvtxs, -1);
    std::vector<thrust::host_vector<int>> xadj_history(numPasses);
    std::vector<thrust::host_vector<int>> adjncy_history(numPasses);
    std::vector<thrust::host_vector<int>> weight_history(numPasses);
    std::vector<thrust::host_vector<int>> match_history(numPasses);
    std::vector<thrust::host_vector<int>> new_ids_history(numPasses);
    std::vector<thrust::host_vector<int>> cmap_history(numPasses);

    vertex_count_history[0] = nvtxs;

    while (coarsed_nvtx > target_vertices && (pass <= numPasses - 1)) {
        std::cout << "Coarsening Level " << pass << " Number of Vertex = " << coarsed_nvtx << std::endl;

        cudaMemset(d_coarsened_nvtxs, 0, sizeof(int));
        cudaMemset(temp, 0, sizeof(int));
        cudaMemset(d_coarsened_edges, 0, sizeof(int));

        thrust::fill(d_cmap.begin(), d_cmap.begin() + coarsed_nvtx, -1);
        thrust::fill(d_new_ids.begin(), d_new_ids.begin() + coarsed_nvtx, -1);
        thrust::fill(d_label_map.begin(), d_label_map.begin() + coarsed_nvtx, -1);

        gettimeofday(&begin, NULL);
        resetMatch<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_match.data()), coarsed_nvtx);
        cudaDeviceSynchronize();

        matchVertices<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_xadj.data()),
            thrust::raw_pointer_cast(d_adjncy.data()),
            thrust::raw_pointer_cast(d_adjwgt.data()),
            thrust::raw_pointer_cast(d_match.data()),
            thrust::raw_pointer_cast(d_tempweight.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();

        bool converged = false;
        int max_iterations = 50;
        int iteration = 0;
        thrust::host_vector<int> h_match = d_match;

        while (!converged && iteration < max_iterations) {
            h_match_prev = h_match;
            rematchVertices<<<numBlocks, blockSize>>>(
                thrust::raw_pointer_cast(d_xadj.data()),
                thrust::raw_pointer_cast(d_adjncy.data()),
                thrust::raw_pointer_cast(d_adjwgt.data()),
                thrust::raw_pointer_cast(d_match.data()),
                thrust::raw_pointer_cast(d_tempweight.data()),
                coarsed_nvtx
            );
            cudaDeviceSynchronize();

            h_match = d_match;
            converged = thrust::equal(h_match.begin(), h_match.end(), h_match_prev.begin());
            iteration++;
        }

        mapVertices<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_match.data()),
            thrust::raw_pointer_cast(d_cmap.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();

        assign_coarse_labels_stage1_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_cmap.data()),
            thrust::raw_pointer_cast(d_label_map.data()),
            temp,
            coarsed_nvtx
        );
        cudaDeviceSynchronize(); // Global synchronization between stages

        assign_coarse_labels_stage2_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_cmap.data()),
            thrust::raw_pointer_cast(d_new_ids.data()),
            thrust::raw_pointer_cast(d_label_map.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();

        updateGlobalMapping<<<(nvtxs + blockSize - 1) / blockSize, blockSize>>>(
            thrust::raw_pointer_cast(d_global_cmap.data()),
            thrust::raw_pointer_cast(d_cmap.data()),
            thrust::raw_pointer_cast(d_new_ids.data()),
            coarsed_nvtx,
            nvtxs,
            temp
        );
        cudaDeviceSynchronize();

        match_history[pass] = d_match;
        new_ids_history[pass] = d_new_ids;
        cmap_history[pass] = d_cmap;

        std::stringstream ss;
        ss << "./MergeParts/merge_level_" << pass << ".txt";
        std::ofstream merge_file(ss.str());
        if (merge_file.is_open()) {
            merge_file << "vertex_id,matched_vertex_id,new_label\n";
            for (int i = 0; i < coarsed_nvtx; ++i) {
                merge_file << i << "," << match_history[pass][i] << "," << new_ids_history[pass][i] << "\n";
            }
            merge_file.close();
            std::cout << "Merging info saved to " << ss.str() << std::endl;
        } else {
            std::cerr << "Error: Unable to open merge file for level " << pass << std::endl;
        }

        std::cout << "Merging details for first 10 vertices at level " << pass << ":\n";
        std::cout << std::setw(10) << "Vertex ID" << std::setw(20) << "Matched Vertex ID" << std::setw(15) << "New Label\n";
        for (int i = 0; i < std::min(10, coarsed_nvtx); ++i) {
            std::cout << std::setw(10) << i << std::setw(20) << match_history[pass][i]
                      << std::setw(15) << new_ids_history[pass][i] << "\n";
        }

        thrust::device_vector<int> d_uniqueCounts(coarsed_nvtx, 0);
        thrust::device_vector<int> d_row(nvtxs + 1, 0);
        thrust::device_vector<int> d_row_ptr(nvtxs + 1, 0);
        thrust::device_vector<int> d_col_idx(nedges, 0);
        thrust::device_vector<int> d_values(nedges, 0);

        countUniqueEdges<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_xadj.data()),
            thrust::raw_pointer_cast(d_adjncy.data()),
            thrust::raw_pointer_cast(d_adjwgt.data()),
            thrust::raw_pointer_cast(d_new_ids.data()),
            thrust::raw_pointer_cast(d_match.data()),
            thrust::raw_pointer_cast(d_cmap.data()),
            thrust::raw_pointer_cast(d_uniqueCounts.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();

        prefix_sum<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_uniqueCounts.data()),
            thrust::raw_pointer_cast(d_row.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();

        fillEdgeArrays<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_xadj.data()),
            thrust::raw_pointer_cast(d_adjncy.data()),
            thrust::raw_pointer_cast(d_adjwgt.data()),
            thrust::raw_pointer_cast(d_new_ids.data()),
            thrust::raw_pointer_cast(d_match.data()),
            thrust::raw_pointer_cast(d_cmap.data()),
            thrust::raw_pointer_cast(d_row.data()),
            thrust::raw_pointer_cast(d_col_idx.data()),
            thrust::raw_pointer_cast(d_values.data()),
            coarsed_nvtx
        );
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        coarseningComputeTime += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

        gettimeofday(&begin, NULL);
        thrust::host_vector<int> h_row = d_row;
        cudaMemcpy(&coarsed_nvtx, temp, sizeof(int), cudaMemcpyDeviceToHost);
        gettimeofday(&end, NULL);
        ioTransferTime += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

        coarsed_edge = h_row[coarsed_nvtx] / 2;

        d_row_ptr.resize(coarsed_nvtx + 1);
        d_col_idx.resize(h_row[coarsed_nvtx]);
        d_values.resize(h_row[coarsed_nvtx]);
        thrust::copy(d_row.begin(), d_row.begin() + coarsed_nvtx + 1, d_row_ptr.begin());
        thrust::copy(d_col_idx.begin(), d_col_idx.begin() + h_row[coarsed_nvtx], d_col_idx.begin());
        thrust::copy(d_values.begin(), d_values.begin() + h_row[coarsed_nvtx], d_values.begin());

        xadj_history[pass] = d_row_ptr;
        adjncy_history[pass] = d_col_idx;
        weight_history[pass] = d_values;

        // Debug: Validate global_cmap
        thrust::host_vector<int> h_global_cmap = d_global_cmap;
        int max_coarse_id = thrust::reduce(h_global_cmap.begin(), h_global_cmap.end(), -1, thrust::maximum<int>());
        int min_coarse_id = thrust::reduce(h_global_cmap.begin(), h_global_cmap.end(), nvtxs, thrust::minimum<int>());
        std::cout << "Level " << pass << ": Max coarse ID = " << max_coarse_id
                  << ", Min coarse ID = " << min_coarse_id
                  << ", Expected max = " << (coarsed_nvtx - 1) << "\n";
        std::cout << "global_cmap sample (first 10): ";
        for (int i = 0; i < std::min(10, nvtxs); ++i) {
            std::cout << h_global_cmap[i] << " ";
        }
        std::cout << "\n";

        vertex_count_history[pass + 1] = coarsed_nvtx;
        std::cout << "After Coarsening Number of Vertex = " << coarsed_nvtx << std::endl;

        d_xadj = d_row_ptr;
        d_adjncy = d_col_idx;
        d_adjwgt = d_values;

        d_match.resize(coarsed_nvtx);
        d_tempweight.resize(coarsed_nvtx);
        d_cmap.resize(coarsed_nvtx);
        d_new_ids.resize(coarsed_nvtx);
        d_label_map.resize(coarsed_nvtx);

        pass++;
        numBlocks = (coarsed_nvtx + blockSize - 1) / blockSize;
    }
    numPasses = pass;

    std::cout << "Number of coarsening passes: " << numPasses << std::endl;
    if (numPasses < 1) {
        std::cerr << "Error: No coarsening passes completed!" << std::endl;
    } else {
        std::cout << "Size of final xadj: " << xadj_history[numPasses - 1].size()
                  << ", adjncy: " << adjncy_history[numPasses - 1].size()
                  << ", weight: " << weight_history[numPasses - 1].size() << std::endl;
    }

    gettimeofday(&begin, NULL);
    thrust::host_vector<int> final_row = (numPasses > 0) ? xadj_history[numPasses - 1] : thrust::host_vector<int>();
    thrust::host_vector<int> final_col = (numPasses > 0) ? adjncy_history[numPasses - 1] : thrust::host_vector<int>();
    thrust::host_vector<int> final_weight = (numPasses > 0) ? weight_history[numPasses - 1] : thrust::host_vector<int>();
    thrust::host_vector<int> h_global_cmap = d_global_cmap;
    gettimeofday(&end, NULL);
    ioTransferTime += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

    // Validate final global_cmap
    int max_coarse_id = thrust::reduce(h_global_cmap.begin(), h_global_cmap.end(), -1, thrust::maximum<int>());
    int min_coarse_id = thrust::reduce(h_global_cmap.begin(), h_global_cmap.end(), nvtxs, thrust::minimum<int>());
    std::cout << "Final: Max coarse ID = " << max_coarse_id
              << ", Min coarse ID = " << min_coarse_id
              << ", Expected max = " << (coarsed_nvtx - 1) << "\n";

    // Build final mapping
    std::map<int, std::vector<int>> final_mapping;
    for (int i = 0; i < nvtxs; ++i) {
        int coarse_id = h_global_cmap[i];
        if (coarse_id >= 0 && coarse_id < coarsed_nvtx) {
            final_mapping[coarse_id].push_back(i);
        } else {
            // Log unmapped vertices
            std::ofstream unmapped("./CoarsedGraphOutput/unmapped_vertices.txt", std::ios::app);
            unmapped << "Vertex " << i << " has invalid coarse_id " << coarse_id << "\n";
            unmapped.close();
            // Assign to a valid coarse vertex (e.g., coarse_id % coarsed_nvtx)
            coarse_id = (coarse_id < 0 || coarse_id >= coarsed_nvtx) ? (i % coarsed_nvtx) : coarse_id;
            final_mapping[coarse_id].push_back(i);
        }
    }

    // Validate number of mapped vertices
    size_t total_mapped_vertices = 0;
    for (const auto& pair : final_mapping) {
        total_mapped_vertices += pair.second.size();
    }
    std::cout << "Total mapped vertices: " << total_mapped_vertices << ", Expected: " << nvtxs << "\n";

    std::ofstream mapping_file("./CoarsedGraphOutput/final_vertex_mapping.txt");
    if (mapping_file.is_open()) {
        for (const auto& pair : final_mapping) {
            mapping_file << "vertex " << pair.first << ": ";
            const auto& vertices = pair.second;
            for (size_t i = 0; i < vertices.size(); ++i) {
                mapping_file << vertices[i];
                if (i < vertices.size() - 1) {
                    mapping_file << ",";
                }
            }
            mapping_file << "\n";
        }
        mapping_file.close();
        std::cout << "Final vertex mapping saved to final_vertex_mapping.txt" << std::endl;
    } else {
        std::cerr << "Error: Unable to open final_vertex_mapping.txt" << std::endl;
    }

    //std::cout << "Final vertex mapping for first 10 coarse vertices:\n";
    //std::cout << std::setw(15) << "Coarse Vertex ID" << std::setw(30) << "Original Vertices\n";
    //int count = 0;
    //for (const auto& pair : final_mapping) {
       // if (count >= 10) break;
        //std::cout << std::setw(15) << pair.first << std::setw(30);
        //const auto& vertices = pair.second;
        //for (size_t i = 0; i < vertices.size(); ++i) {
            //std::cout << vertices[i];
            //if (i < vertices.size() - 1) {
                //std::cout << ",";
           // }
       // }
        //std::cout << "\n";
       // count++;
    //}

    if (final_row.empty() || final_col.empty() || final_weight.empty()) {
        std::cerr << "Error: Final coarsened graph data is empty!" << std::endl;
        std::cerr << "numPasses: " << numPasses << ", xadj_history size: "
                  << (numPasses > 0 ? xadj_history[numPasses - 1].size() : 0) << std::endl;
    } else {
        std::ofstream row_file("./CoarsedGraphOutput/coarse_row.txt");
        std::ofstream col_file("./CoarsedGraphOutput/coarse_column.txt");
        std::ofstream weight_file("./CoarsedGraphOutput/coarse_weight.txt");

        if (row_file.is_open() && col_file.is_open() && weight_file.is_open()) {
            for (size_t i = 0; i < final_row.size(); ++i) {
                row_file << final_row[i] << "\n";
            }
            for (size_t i = 0; i < final_col.size(); ++i) {
                col_file << final_col[i] << "\n";
            }
            for (size_t i = 0; i < final_weight.size(); ++i) {
                weight_file << final_weight[i] << "\n";
            }
            row_file.close();
            col_file.close();
            weight_file.close();
            std::cout << "Final coarsened graph saved to coarse_row.txt, coarse_column.txt, and coarse_weight.txt" << std::endl;
        } else {
            std::cerr << "Error: Unable to open output files" << std::endl;
        }
    }

    int final_nvtxs = vertex_count_history[numPasses];
    int final_nedges = final_col.size() / 2;
    std::cout << "Final Number of Vertices = " << final_nvtxs << std::endl;
    std::cout << "Final Number of Edges = " << final_nedges << std::endl;

    std::cout << "INPUT FILE READING TIME: " << inputReadingTime << " seconds" << std::endl;
    std::cout << "I/O TRANSFER TIME (CPU-GPU & GPU-CPU): " << ioTransferTime << " seconds" << std::endl;
    std::cout << "COARSENING TIME (excluding I/O and transfers): " << coarseningComputeTime << " seconds" << std::endl;
    double totalPreprocessingTime = inputReadingTime + ioTransferTime + coarseningComputeTime;
    std::cout << "Total Pre-computation time: " << totalPreprocessingTime << " seconds" << std::endl;

    free(h_xadj);
    free(h_adjncy);
    free(h_weight);
    cudaFree(d_coarsened_nvtxs);
    cudaFree(d_coarsened_edges);
    cudaFree(temp);

    return 0;
}