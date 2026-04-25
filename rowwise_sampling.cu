/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

//// sampling kernel call function in line 1236


#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/tensordispatch.h>

#include <cstdint>
#include <iostream>
#include <numeric>

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./dgl_cub.cuh"
#include "./utils.h"

#include "../../array/cuda/global_array.h"

using namespace dgl::cuda;
using namespace dgl::aten::cuda;
using TensorDispatcher = dgl::runtime::TensorDispatcher;

namespace dgl {
namespace aten {
namespace impl {

namespace {

constexpr int BLOCK_SIZE = 128;
constexpr int METIS_BLOCK_SIZE = 128;
constexpr int64_t SELECT_NEB_SIZE = 64;
// constexpr int SHARED_MEM_SIZE = 128;
int flag = 0;
// int is_sampled_flag = 0;
int64_t* d_part_array;
// int64_t* is_sampled_array;
float sampling_time = 0.0;
/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
      static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}
/**
 row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
r The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}



template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel1(    //original code
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* d_part_array) {

//--------------sampled the vertices from same partition---------------------------//
//--------------skip the vertices which is sampled earlier------------------------//
 // __global__ void _CSRRowWiseSampleUniformKernel1(
 //  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
 //  const IdType* const in_rows, const IdType* const in_ptr,
 //  const IdType* const in_index, const IdType* const data,
 //  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
 //  IdType* const out_idxs, int64_t* d_part_array, int64_t* is_sampled_array) {
 // we assign one warp per row
  assert(blockDim.x == METIS_BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  // printf("num_rows: %ld, blockIdx.x: %d, TILE_SIZE: %d\n", num_rows, blockIdx.x, TILE_SIZE);
  // if (blockIdx.x==0 && threadIdx.x == 0)
  // {
  //   // printf("Kishan define function that passed d_part_array and 1st data is :%ld\n",d_part_array[0]);
  //   // printf("out_row: %ld\n", out_row);
  // }
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    // if(blockIdx.x ==0 && threadIdx.x ==0)
    // {
      // printf("row: %ld, in_row_start: %ld, deg: %ld, out_row_start: %ld\n", row, in_row_start, deg, out_row_start);
    // }

    //-------------original code-----------------//
    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += METIS_BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } 

    //--------------------skip vertex which is already sampled--------//
    // if (deg <= num_picks) {
    //   // just copy row when there is not enough nodes to sample.
    //   for (int idx = threadIdx.x; idx < deg; idx += METIS_BLOCK_SIZE) {
    //     const IdType in_idx = in_row_start + idx;
    //     // out_rows[out_row_start + idx] = row;
    //     // out_cols[out_row_start + idx] = in_index[in_idx];
	// if (is_sampled_array[in_index[in_idx]] == 0)
	// {
	  // // out_idxs[out_row_start + idx] = in_index[in_idx];
	  // out_rows[out_row_start + idx] = row;
	  // out_cols[out_row_start + idx] = in_index[in_idx];
    //       out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
	  // is_sampled_array[in_index[in_idx]] = 1; // mark as sampled
	// }
	// // else
	// // {
	// //   // out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
	// //   is_sampled_array[in_index[in_idx]] = 1; // mark as sampled
	// // }
    //     // out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
    //   }
    // } 

    else 
  {
      // Shared memory to store counts for each value and their index
      // extern __shared__ int counts[];
      extern __shared__ int index_array[];
      extern __shared__ int non_part_vertex[];
      // __shared__ int index_array[10];
      __shared__ int maxCount;
      __shared__ int maxValue;
      __shared__ int pointer_position;
      __shared__ int non_part_vertex_pointer;
      // int pointer_position;
      // __shared__ int neighbours[METIS_BLOCK_SIZE];
      // __shared__ int part_id[METIS_BLOCK_SIZE];
      // __shared__ int neighbours[SHARED_MEM_SIZE];
      // __shared__ int part_id[SHARED_MEM_SIZE];

      // initilize 0 
      // if ( deg < SHARED_MEM_SIZE)
      if ( deg < METIS_BLOCK_SIZE)
      {
      	__shared__ int neighbours[METIS_BLOCK_SIZE];
      	__shared__ int part_id[METIS_BLOCK_SIZE];
      	__shared__ int neb_deg[METIS_BLOCK_SIZE];   //neighbors degree for choose high degree node
        if(threadIdx.x < deg)
	// for(int i = threadIdx.x; i < deg; i += METIS_BLOCK_SIZE)
        {
          // int neb_data = in_index[in_row_start + i];
	  
          int neb_data = in_index[in_row_start + threadIdx.x];
          // neighbours[threadIdx.x] = in_index[in_row_start + threadIdx.x];
          neighbours[threadIdx.x] = neb_data;
          part_id[threadIdx.x] = d_part_array[neb_data];
	  neb_deg[threadIdx.x] = in_ptr[neb_data + 1] - in_ptr[neb_data];
          // neighbours[i] = neb_data;
          // part_id[i] = d_part_array[neb_data];

          // part_id[threadIdx.x] = d_part_array[in_index[in_row_start + threadIdx.x]];
        }
        // if (threadIdx.x < num_picks) {
        //   counts[threadIdx.x] = 0;
        // }

        if(threadIdx.x == 0)
        {
          maxValue = d_part_array[row]; //source node part id
          // counts[maxValue] = 0;
          pointer_position = 0;
          non_part_vertex_pointer = 0;
        }
        __syncthreads();

        // for(int kk= threadIdx.x; kk < deg; kk+=BLOCK_SIZE)
        // // if(threadIdx.x < deg)
        // {
        //   int val = part_id[kk];
        //   if( val == maxValue)
        //   {
        //     atomicAdd(&counts[val], 1);
        //   }
        //   if (counts[val] >= num_picks)
        //   {
        //     break;
        //   }
        // }

        // if(threadIdx.x == 0)
        // {
        // for( int kk = threadIdx.x; kk < deg; kk += BLOCK_SIZE)
        // // for (int kk = 0; kk < deg; kk++)
        // {
        //
        //   // int val = d_part_array[in_index[in_row_start + kk]];
        //   int val = part_id[kk];
        //   if( val == maxValue && pointer_position < num_picks)
        //   {
        //     atomicAdd(&counts[val], 1);
        //     // counts[val] = counts[val] + 1;
        //     int array_pos = atomicAdd(&pointer_position, 1);
        //     index_array[array_pos] = kk;
        //     // index_array[pointer_position] = kk;
        //     // pointer_position++;
        //   }
        //   else if(non_part_vertex_pointer < num_picks)
        //   {
        //     int non_p_arr_ptr = atomicAdd(&non_part_vertex_pointer, 1);
        //     non_part_vertex[non_p_arr_ptr] = kk;
        //     // non_part_vertex[non_part_vertex_pointer] = kk;
        //     // non_part_vertex_pointer++;
        //   }
        //   // if (counts[val] >= num_picks && pointer_position >= num_picks && non_part_vertex_pointer >= num_picks)
        //   // if (pointer_position >= num_picks && non_part_vertex_pointer >= num_picks)
        //   // {
        //     // break;
        //   // }
        //   // atomicAdd(&counts[val], 1);
        // }
        // // } //threadIdx.x == 0

        // Find maximum count and corresponding value
        // if (threadIdx.x == 0) {
        //   maxCount = 0;
        //   // pointer_position = 0; 
        //   // maxValue = d_part_array[row];
        //   maxCount = counts[maxValue];
        //   // printf("maxCount: %d, maxValue: %d, row: %ld\n",maxCount, maxValue, d_part_array[row]);
        // }
        // __syncthreads();
          // Find indices of maximum repeated value
        if (threadIdx.x == 0)
        {
          // int non_part_vertex_pointer=0;
          for(int kk = 0; kk < deg; kk++)
          // for(int kk = threadIdx.x; kk < deg; kk += METIS_BLOCK_SIZE)
          {
            if(part_id[kk] == maxValue && pointer_position < num_picks)
            // if(part_id[threadIdx.x] == maxValue && pointer_position < num_picks)
            // if(d_part_array[in_index[in_row_start + kk]] == maxValue && pointer_position < num_picks)
            {
              // index_array[pointer_position] = threadIdx.x;
	      // int loc = atomicAdd(&pointer_position, 1);
              // index_array[loc] = kk;
              index_array[pointer_position] = kk;
              pointer_position++;
            }
	    else if(part_id[kk] != maxValue && non_part_vertex_pointer < num_picks)
            {
              // non_part_vertex[non_part_vertex_pointer] = threadIdx.x;
	      // int loc = atomicAdd(&non_part_vertex_pointer, 1);
              // non_part_vertex[loc] = kk;
              non_part_vertex[non_part_vertex_pointer] = kk;
              non_part_vertex_pointer++;
            }
            if( pointer_position >= num_picks-1 && non_part_vertex_pointer >= num_picks-1)
            {
              break;
            }
          }
          maxCount = pointer_position;
        } //threadidx.x ==0
          // generate permutation list via reservoir algorithm
        for (int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE) {
          out_idxs[out_row_start + idx] = idx;
        }
        __syncthreads();
        if (num_picks < maxCount)
        {
          for (int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
          {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = neighbours[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
        else {
          // copy maximum same partition vertex avalible 
          for (int idx = threadIdx.x; idx < maxCount; idx += METIS_BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = neighbours[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
          // int remining_size = num_picks - maxCount;
          // printf("remining size: %d maxCount: %d\n",remining_size,maxCount);
          for (int idx = threadIdx.x + maxCount; idx < num_picks; idx += METIS_BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = neighbours[non_part_vertex[idx - maxCount]];
            // int counter = 0;
            // for(int i = 0; i < deg && counter < remining_size; i++)
            // {
            //   int flag = 0;
            //   // printf("Bid: %d, threadIdx: %d, row: %d \n", blockIdx.x, threadIdx.x,row);
            //   for (int j = 0; j < maxCount; j++)
            //   {
            //     if ( neighbours[i] == neighbours[index_array[j]])
            //     {
            //       flag = 1;
            //       break;
            //     }
            //   }
            //   if (flag == 0)
            //   {
            //     out_cols[out_row_start + idx] = neighbours[i];
            //     counter++;
            //   }
            // }
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
      }
      else
    {
        // if (threadIdx.x < num_picks) {
        //   counts[threadIdx.x] = 0;
        // }
        // __syncthreads();

        //count the total repeated values of same partition
        if( threadIdx.x == 0)
        {
          maxValue = d_part_array[row];
          pointer_position = 0;
          non_part_vertex_pointer = 0;
          // counts[maxValue] = 0;
          // maxCount = 0;
        }
        // for( int kk = threadIdx.x; kk < deg; kk += BLOCK_SIZE)
        // {
        //
        //   int val = d_part_array[in_index[in_row_start + kk]];
        //   if( val == maxValue)
        //   {
        //     atomicAdd(&counts[val], 1);
        //   }
        //   if (counts[val] >= num_picks)
        //   {
        //     break;
        //   }
        //   // atomicAdd(&counts[val], 1);
        // }
        // __syncthreads();
        // if(threadIdx.x == 0)
        {
        for( int kk = threadIdx.x; kk < deg; kk += BLOCK_SIZE)
        // for (int kk = 0; kk < deg; kk++)
        {

          int val = d_part_array[in_index[in_row_start + kk]];
          if( val == maxValue && pointer_position < num_picks)
          {
            // atomicAdd(&counts[val], 1);
            // counts[val] = counts[val] + 1;
            int array_pos = atomicAdd(&pointer_position, 1);
            // index_array[pointer_position] = in_row_start + kk;
            index_array[array_pos] = in_row_start + kk;
            // pointer_position++;
          }
          else if(non_part_vertex_pointer < num_picks)
          {
            int non_p_arr_ptr = atomicAdd(&non_part_vertex_pointer, 1);
            non_part_vertex[non_p_arr_ptr] = in_row_start + kk;
            // non_part_vertex[non_part_vertex_pointer] = in_row_start + kk;
            // non_part_vertex_pointer++;
          }
          if (pointer_position >= num_picks-1 && non_part_vertex_pointer >= num_picks-1)
          {
            break;
          }
          // atomicAdd(&counts[val], 1);
        }
          maxCount = pointer_position;
        } //threadIdx.x == 0
        // __syncthreads();

        // Find maximum count and corresponding value
        // if (threadIdx.x == 0) {
          // maxCount = 0;
          // pointer_position = 0; 
          // maxCount = counts[maxValue];
        // }
        __syncthreads();

        // Find indices of maximum repeated value
        // if (threadIdx.x == 0)
        // {
        //   int non_part_vertex_pointer=0;
        //   for(int kk = 0; kk < deg; kk++)
        //   {
        //     if(d_part_array[in_index[in_row_start + kk]] == maxValue && pointer_position < num_picks)
        //     {
        //       index_array[pointer_position] = in_row_start + kk;
        //       // index_array[pointer_position] = in_index[in_row_start + kk];
        //       pointer_position++;
        //     }
        //     if(d_part_array[in_index[in_row_start + kk]] != maxValue && non_part_vertex_pointer < num_picks)
        //     {
        //       non_part_vertex[non_part_vertex_pointer] = in_row_start + kk;
        //       non_part_vertex_pointer++;
        //     }
        //     if( pointer_position >= num_picks || non_part_vertex_pointer >= num_picks)
        //
        //     // if( pointer_position >= num_picks)
        //     {
        //       break;
        //     }
        //   }
        // }

        // generate permutation list via reservoir algorithm
        for (int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE) {
          out_idxs[out_row_start + idx] = idx;
        }
        __syncthreads();

        // copy all vertex if avalible of same partition
        if (num_picks < maxCount)
        {
          for (int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
          {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = in_index[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
        else {
          // copy maximum same partition vertex avalible 
          for (int idx = threadIdx.x; idx < maxCount; idx += METIS_BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = in_index[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
          // int remining_size = num_picks - maxCount;
          // printf("remining size: %d maxCount: %d\n",remining_size,maxCount);
          for (int idx = threadIdx.x + maxCount; idx < num_picks; idx += METIS_BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = in_index[non_part_vertex[idx - maxCount]];
            // int counter = 0;
            // for(int i = 0; i < deg && counter < remining_size; i++)
            // {
            //   int flag = 0;
            //   // printf("Bid: %d, threadIdx: %d, row: %d \n", blockIdx.x, threadIdx.x,row);
            //   for (int j = 0; j < maxCount; j++)
            //   {
            //     if ( in_index[i + in_row_start] == in_index[index_array[j]])
            //     {
            //       flag = 1;
            //       break;
            //     }
            //
            //   }
            //   if (flag == 0)
            //   {
            //     out_cols[out_row_start + idx] = in_index[i + in_row_start];
            //     counter++;
            //   }
            // }
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
      }
    }
    out_row += 1;
  }
}


template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel_sort(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* d_part_array
  , int64_t num_edges, int64_t num_vertex, int64_t d_part_array_size
  ) {

//--------------sampled the vertices from same partition---------------------------//
//--------------skip the vertices which is sampled earlier------------------------//
  assert(blockDim.x == METIS_BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
	
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t in_row_end = in_ptr[row + 1];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    //-------------original code-----------------//
    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += METIS_BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      } //for loop end
    } // end if
    else
    {
      for (int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = d_part_array[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx; 
	}
    }
    out_row += 1;
  }
}
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel_sort_old(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* d_part_array
  , int64_t num_edges, int64_t num_vertex, int64_t d_part_array_size
  ) {

//--------------sampled the vertices from same partition---------------------------//
//--------------skip the vertices which is sampled earlier------------------------//
  assert(blockDim.x == METIS_BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
	
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t in_row_end = in_ptr[row + 1];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    // if(blockIdx.x ==0 && threadIdx.x ==0)
    // {
      // printf("row: %ld, in_row_start: %ld, deg: %ld, out_row_start: %ld\n", row, in_row_start, deg, out_row_start);
    // }

    //-------------original code-----------------//
    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += METIS_BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      } //for loop end
    } // end if
    else
    {
	// for(int idx = threadIdx.x + in_row_start; idx < num_picks; idx += METIS_BLOCK_SIZE)
	// int start_part_loc = 0;
	// int total_remming_vertex = 0;
	// int flag_loc = 0;
	// int select_ptr = 0;
	// int select_non_part_ptr = 0;
	extern __shared__ int64_t neb_id_list[];
	// __shared__ int64_t neb_id_list[SELECT_NEB_SIZE]; // double the size to store both vertex and index
      	__shared__ int64_t non_part_neb[SELECT_NEB_SIZE];
	// extern __shared__ int non_part_neb[];
	// extern __shared__ int final_picked_vertex[];
	__shared__ int select_ptr; // pointer to store the selected vertices
	__shared__ int copy_ptr;
	// __shared__ int seed_node_part; // source node part id
	__shared__ int select_non_part_ptr; // pointer to store the selected non-partition vertices
	// __shared__ int new_num_picks; // number of picks to be made
	// if (threadIdx.x == 0)
	// {
	// 	select_ptr = 0; // pointer to store the selected vertices
	// 	select_non_part_ptr = 0; // pointer to store the selected non-partition vertices
	// 	seed_node_part = d_part_array[ row + num_edges]; //source node part id	
	// }
	// __syncthreads();
	if(threadIdx.x == 0)
	{
	select_ptr = 0; // pointer to store the selected vertices
	select_non_part_ptr = 0; // pointer to store the selected non-partition vertices
	copy_ptr = num_picks / 2;
	// new_num_picks = num_picks / 2; // double the picks to store both vertex and index
	int seed_node_part = d_part_array[ row + num_edges]; //source node part id	
	// printf("row: %ld, seed_node_part: %d\n", row, seed_node_part);
	for(int64_t i = in_row_start; i < in_row_end; i++)
	// for(int i = in_row_start; i < in_row_end && select_ptr < num_picks; i++)
	// for(int i = threadIdx.x + in_row_start; i < in_row_end; i += METIS_BLOCK_SIZE)
	{
	  int64_t Neb = d_part_array[i]; 
	  int64_t Neb_part = d_part_array[Neb + num_edges];
	  // if(d_part_array[Neb + num_edges] == seed_node_part)
	  // if(Neb_part == seed_node_part && pointer_position < num_picks)
	  if(Neb_part == seed_node_part && select_ptr < copy_ptr)   //single thread condition
	  // if(Neb_part == seed_node_part && select_ptr < num_picks)   //single thread condition
	  // if(Neb_part == seed_node_part)
	  {
	    // printf("loc_start: %ld, d_part_array[in_index[loc_start] + num_edges]: %d\n", loc_start, d_part_array[in_index[loc_start] + num_edges]);
	    // int save_loc = atomicAdd(&select_ptr, 1);
	    // int save_loc = atomicAdd(&pointer_position, 1);
	    // if(select_ptr < num_picks)
	    // {
	    // 	int loc = atomicAdd(&select_ptr, 1);
	    // 	neb_list[loc] = Neb;
	    // }
	    neb_id_list[select_ptr] = Neb;
	    // neb_id_list[select_ptr + num_picks] = i; // store the index of the vertex
	    select_ptr++;
	    // out_rows[out_row_start + select_ptr] = row;
	    // out_cols[out_row_start + select_ptr] = Neb;
	    // out_idxs[out_row_start + select_ptr] = data ? data[i] : i;
	    // select_ptr++;
	    // if (flag_loc == 0)
	    // {
	    //   flag_loc = 1;
	    //   start_part_loc = i;
	    //   total_remming_vertex = in_row_end - i;
	    // }
	    // start_part_loc = i;
	    // total_remming_vertex = in_row_end - i;
	    // break;
	  }  //end if
	  // else if (Neb_part != seed_node_part && pointer_position_non_part < num_picks)
	  // else if (select_non_part_ptr < copy_ptr)
	  else if (select_non_part_ptr < num_picks)
	  // else if (Neb_part != seed_node_part)
	  {
	     // if (select_non_part_ptr < num_picks)
	     // // if (select_non_part_ptr < num_picks)
	     // {
	     // 	int loc = atomicAdd(&select_non_part_ptr, 1);
	     // 	non_part_neb[loc] = Neb;
	     // }
	     // neb_id_list[select_non_part_ptr + num_picks] = Neb; // store the index of the vertex
	     // neb_id_list[select_non_part_ptr + (num_picks *3)] = i; // store the index of the vertex
	     // select_non_part_ptr++;
	     non_part_neb[select_non_part_ptr] = Neb;
	     // non_part_neb[select_non_part_ptr + num_picks] = i;
	     select_non_part_ptr++;
	  }  //end else
	  // if (pointer_position == num_picks)
	  if (select_ptr == copy_ptr && select_non_part_ptr == num_picks)
	  // if (select_ptr == num_picks && select_non_part_ptr == num_picks)
	  {
		  break;
	  }
	} //end for
	} //end thread if
	__syncthreads();
	// if(threadIdx.x == 0)
	// {
	// 	printf("tid: %d, bid: %d, row: %ld, select_ptr: %d, select_non_part_ptr: %d, num_picks: %ld\n", threadIdx.x, blockIdx.x, row, select_ptr, select_non_part_ptr, num_picks);
	// }
	// if (select_ptr < num_picks)
	// {
	// 	int remining_size = num_picks - select_ptr;
	// 	for(int idx = threadIdx.x; idx < remining_size; idx += METIS_BLOCK_SIZE)
	// 	{	 
	// 	       	out_rows[out_row_start + idx + select_ptr] = row;
	// 		out_cols[out_row_start + idx + select_ptr] = non_part_neb[idx];
	// 		out_idxs[out_row_start + idx + select_ptr] = data ? data[non_part_neb[idx + num_picks]] : non_part_neb[idx + num_picks];
	// 		// neb_list[idx + select_ptr] = d_part_array[in_row_start + idx];
	// 		// neb_list[idx + select_ptr] = non_part_neb[idx];
	// 	}
	// } //end if
	// if (num_picks <= select_ptr)
	// {
	// 	for(int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
	// 	{
	// 		final_picked_vertex[idx] = neb_list[idx];
	// 	}
	// }
	// else
	// {
	// 	for(int idx = threadIdx.x; idx < select_ptr; idx += METIS_BLOCK_SIZE)
	// 	{
	// 		final_picked_vertex[idx] = neb_list[idx];
	// 	}
	// 	int remining_size = num_picks - select_ptr;
	// 	for(int idx = threadIdx.x; idx < remining_size; idx += METIS_BLOCK_SIZE)
	// 	{
	// 		final_picked_vertex[idx + select_ptr] = non_part_neb[idx];
	// 	}
	// }
	// __syncthreads();
	// for( int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
	// {
	//   // copy the indices which is alredy sorted based on partition and degree
	//   const IdType in_idx = in_row_start + final_picked_vertex[idx];
	//   out_rows[out_row_start + idx] = row;
	//   out_cols[out_row_start + idx] = final_picked_vertex[idx];
	//   out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
	// }
	// for( int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
	// {
	//   // copy the indices which is alredy sorted based on partition and degree
	//   const IdType in_idx = in_row_start + idx;
	//   out_rows[out_row_start + idx] = row;
	//   out_cols[out_row_start + idx] = neb_list[idx];
	//   out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
	// }
	if (copy_ptr == select_ptr && num_picks == select_non_part_ptr)
	// if (num_picks == select_ptr && num_picks == select_non_part_ptr)
	{	
		// if(threadIdx.x == 0)
		// {
		// 	copy_ptr = num_picks / 2;
		// }
		// __syncthreads();
		// for(int idx = threadIdx.x; idx < copy_ptr; idx += METIS_BLOCK_SIZE)
		for(int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
		{ 
		  // copy the indices which is alredy sorted based on partition and degree
		  const IdType in_idx = in_row_start + idx;
		  out_rows[out_row_start + idx] = row;
		  // if (idx < copy_ptr)
		  // // if (idx < num_picks/2)
		  // {
		  //   out_cols[out_row_start + idx] = neb_id_list[idx];
		  //   // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		  //   out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		  // }
		  // else
		  // {
		  //   out_cols[out_row_start + idx] = non_part_neb[idx - copy_ptr];
		  //   // out_cols[out_row_start + idx] = neb_id_list[idx + num_picks - copy_ptr];
		  //   // out_cols[out_row_start + idx] = d_part_array[idx + in_row_start];
		  //   // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx + (num_picks * 2) - copy_ptr]] : neb_id_list[num_picks + idx + (num_picks * 2) - copy_ptr];
		  //   out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		  // }
		  // out_cols[out_row_start + idx] = neb_id_list[idx];
		  // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		  out_cols[out_row_start + idx] = d_part_array[idx + in_row_start];
		  out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		}
		// for(int idx = threadIdx.x; idx < (num_picks - copy_ptr); idx += METIS_BLOCK_SIZE)
		// { 
		//   // copy the indices which is alredy sorted based on partition and degree
		//   // const IdType in_idx = in_row_start + idx;
		//   out_rows[out_row_start + idx + copy_ptr] = row;
		//   // out_cols[out_row_start + idx + copy_ptr] = neb_id_list[idx + (num_picks)];
		//   out_cols[out_row_start + idx + copy_ptr] = non_part_neb[idx];
		//   // out_idxs[out_row_start + idx + copy_ptr] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		//   // out_cols[out_row_start + idx + copy_ptr] = d_part_array[idx + in_row_start];
		//   out_idxs[out_row_start + idx + copy_ptr] = data ? data[idx] : idx;
		// }
	}
	else
	{
		// for(int idx = threadIdx.x; idx < num_picks; idx += METIS_BLOCK_SIZE)
		// { 
		//   // copy the indices which is alredy sorted based on partition and degree
		//   // const IdType in_idx = in_row_start + idx;
		//   out_rows[out_row_start + idx] = row;
		//   if (idx < select_ptr)
		//   {
		//     out_cols[out_row_start + idx] = neb_id_list[idx];
		//     // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		//     out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		//   }
		//   else
		//   {
		//     out_cols[out_row_start + idx] = neb_id_list[idx + (num_picks * 2) - select_ptr];
		//     // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx + (num_picks * 2) - select_ptr]] : neb_id_list[num_picks + idx + (num_picks * 2) - select_ptr];
		//     out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		//   }
		//   // out_cols[out_row_start + idx] = neb_id_list[idx];
		//   // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		//   // out_cols[out_row_start + idx] = d_part_array[idx + in_row_start];
		//   // out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		// }

		for(int idx = threadIdx.x; idx < select_ptr; idx += METIS_BLOCK_SIZE)
		{ 
		  // copy the indices which is alredy sorted based on partition and degree
		  const IdType in_idx = in_row_start + idx;
		  out_rows[out_row_start + idx] = row;
		  // out_cols[out_row_start + idx] = d_part_array[in_idx];;
		  // out_cols[out_row_start + idx] = neb_id_list[idx];
		  // out_idxs[out_row_start + idx] = data ? data[neb_id_list[num_picks + idx]] : neb_id_list[num_picks + idx];
		  out_cols[out_row_start + idx] = d_part_array[idx + in_row_start];
		  out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		}
		int remining_size = num_picks - select_ptr;
		for(int idx = threadIdx.x; idx < remining_size; idx += METIS_BLOCK_SIZE)
		{ 
		  // copy the indices which is alredy sorted based on partition and degree
		  const IdType in_idx = in_row_start + idx;
		  out_rows[out_row_start + idx + select_ptr] = row;
		  // out_cols[out_row_start + idx + select_ptr] = neb_id_list[idx + num_picks];
		  out_cols[out_row_start + idx + select_ptr] = non_part_neb[idx];
		  // out_cols[out_row_start + idx + select_ptr] = d_part_array[in_idx];
		  // out_idxs[out_row_start + idx + select_ptr] = data ? data[neb_id_list[idx + (num_picks * 2) + num_picks]] : neb_id_list[idx + (num_picks * 2) + num_picks];
		  // out_cols[out_row_start + idx] = d_part_array[idx + in_row_start];
		  out_idxs[out_row_start + idx] = data ? data[idx] : idx;
		}
	}
    }
    out_row += 1;
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZgE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] =
          data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}

}  // namespace

///////////////////////////// CSR sampling //////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 265\n")
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  constexpr int TILE_SIZE = BLOCK_SIZE / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  device->FreeWorkspace(ctx, out_ptr);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("default cuda sapmling time: %.6f\n", milliseconds/1000);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray part_array, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  size_t size = part_array->shape[0];
  // size_t size1 = part_array->shape[1];
  // printf("size of part_array: %ld\n", size);

  // printf("data from _CSRRowWiseSamplingUniform1 function line 448");
  if(flag == 0)
  {
    // printf("flag activated");
    flag = 1;
   int64_t* parts_array = static_cast<int64_t*>(part_array->data);
   // int64_t* next_data = static_cast<int64_t*>(part_array->data);
   // printf("data in next_data: ");
   // for(int i=0; i<3; i++)
   // {
	   // printf("%ld ", next_data[i]);
   // }

    // allocate gpu memory
    // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
    cudaMalloc(&d_part_array, size * sizeof(int64_t));

    cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
    // printf("\nCudamemcpy called\n");
    // flag = 1;
  }
  // if(is_sampled_flag == 0)
  // {
  //   // printf("flag activated");
  //   flag = 1;
  //   // allocate gpu memory
  //   // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  //   cudaMalloc(&is_sampled_array, size * sizeof(int64_t));
  //   cudaMemset(is_sampled_array, 0, size * sizeof(int64_t));

  //   // cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  //   // printf("\nCudamemcpy called\n");
  //   // flag = 1;
  // }

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;

  int64_t num_edges = mat.indices->shape[0];
  int64_t num_nodes = mat.indptr->shape[0];
  printf("num_edges: %ld\n", num_edges);	
  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int TILE_SIZE = METIS_BLOCK_SIZE / METIS_BLOCK_SIZE;
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(METIS_BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    // const dim3 block(BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //     (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
    //     stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //     data, out_ptr, out_rows, out_cols, out_idxs);
    //
    const dim3 block(METIS_BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    ////-------------------------only metis partition-----------------------------
    // CUDA_KERNEL_CALL(
    //   (_CSRRowWiseSampleUniformKernel1<IdType, TILE_SIZE>), grid, block, num_picks,
    //   stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //   data, out_ptr, out_rows, out_cols, out_idxs, d_part_array
    //   // ,is_sampled_array
    //   );
    ////------------------------only metis partition-----------------------------

    ////-----------------------sorted based on partition and degree----------------
     CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel_sort<IdType, TILE_SIZE>), grid, block, 0, 
      // (_CSRRowWiseSampleUniformKernel_sort<IdType, TILE_SIZE>), grid, block, (num_picks*4*sizeof(int64_t)), 
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array
    	    , num_edges, num_nodes, size
      );
   	////-----------------------sorted based on partition and degree----------------

  }
  device->FreeWorkspace(ctx, out_ptr);
  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  sampling_time += milliseconds/1000;
  printf("metis cuda sapmling time %.6f\n", sampling_time);
  // cudaFree(d_part_array);samplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsamplingsampling

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  // std::cout
  // std::cout<<"new_len: "<<new_len<<"\n";
  // std::cout<<"picked_row: "<<picked_row<<"\n";
  // std::cout<<"picked_col: "<<picked_col<<"\n";
  // std::cout<<"picked_idx: "<<picked_idx<<"\n";

  // printf("size of mat.num_rows: %d\n", mat.num_rows);
  // printf("size of mat.num_cols: %d\n", mat.num_cols);
  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  // size_t size = part_array->shape[0];
  // int64_t* parts_array = static_cast<int64_t*>(part_array->data);

  // printf("data from _CSRRowWiseSamplingUniform2 function line 448");
  // if(flag == 0)
  // {
  // printf("flag activated");
  // flag = 1;
  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  // printf("Cudamemcpy called");
  // flag = 1;
  // }
  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int TILE_SIZE = BLOCK_SIZE / BLOCK_SIZE;
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
        stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
        data, out_ptr, out_rows, out_cols, out_idxs);
    //
    // const dim3 block(METIS_BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //   (_CSRRowWiseSampleUniformKernel1<IdType, TILE_SIZE>), grid, block, num_picks,
    //   stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //   data, out_ptr, out_rows, out_cols, out_idxs, d_part_array);

  }
  device->FreeWorkspace(ctx, out_ptr);
  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  sampling_time += milliseconds/1000;
  printf("default cuda sapmling time %.6f\t \n", sampling_time);
  // cudaFree(d_part_array);

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  // std::cout<<"U2 new_len: "<<new_len<<"\n";
  // std::cout<<"U2 picked_row: "<<picked_row<<"\n";
  // std::cout<<"U2 picked_col: "<<picked_col<<"\n";
  // std::cout<<"U2 picked_idx: "<<picked_idx<<"\n";


  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}


template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {

  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << part_array[i] << " ";
  // }

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const NDArray& parts_array, const bool replace) {


  // size_t size = parts_array->shape[0];
  // IdArray* part_array = static_cast<IdArray*>(parts_array->data);
  //
  // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));
  //
  // cudaMemcpy(d_part_array, part_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);

  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << "data :" << part_array[i] << " ";
  // }
  // dgl::IdArray part_array = dgl::aten::AsIdArray(parts_array);
  // dgl::IdArray part_array = dgl::IdArray::FromDLPack(parts_array.ToDLPack());

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform1<XPU, IdType>(
      mat, rows, num_picks, parts_array, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform2<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}


template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, const NDArray& , bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, const NDArray& , bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
} // namespace dgl
