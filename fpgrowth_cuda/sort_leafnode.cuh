#pragma once
#include "data_struct.cuh"

__device__ int compare_prefix(unsigned int *prefix, int i, int j, size_t real_size){
	for (int t = 0; t < real_size; t++){
		if (prefix[i*real_size + t] > prefix[j*real_size + t])
			return 1;
		else if (prefix[i*real_size + t] < prefix[j*real_size + t])
			return -1;
	}
	return 0;
}

__device__ void swapleaf(Node *leaf, int x, int y){
	int temp = leaf[x].id;
	leaf[x].id = leaf[y].id;
	leaf[y].id = temp;

	temp = leaf[x].count;
	leaf[x].count = leaf[y].count;
	leaf[y].count = temp;
}
__global__ void parallel_sort_leaf(
	int n, int m, bool mode,//0-偶数步，1-奇数步
	unsigned int *prefix, size_t real_size,
	Node*leaf
	){
	int i = threadIdx.x +blockDim.x*blockIdx.x;
	if ((i + 1) < n){
		if (i % 2 == (int)mode){
			//printf("i=%d ", i);
			if (leaf[i + 1].count == 0)
				return;
			if (leaf[i].count == 0){
				swapleaf(leaf, i, i + 1);
				//printf("%d,%d\n", i,i+1);
				return;
			}

			int flag = compare_prefix(prefix, leaf[i].id, leaf[i + 1].id, real_size);
			//printf("%d\n", flag);
			if (flag == 1){
				swapleaf(leaf, i, i + 1);
			}
			else if (flag == 0){
				leaf[i].count += leaf[i + 1].count;
				leaf[i + 1].count = 0;
			}
		}
	}
}

__global__ void parallel_find_size(int *result, Node*leaf, int n){
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	*result = n;
	if (idx < n){
		//printf("%d,%d\n", idx, *result);
		if (idx == 0){
			if (leaf[idx].count == 0)
				*result = idx;
		}
		else{
			if (leaf[idx].count == 0 && leaf[idx - 1].count != 0)
				*result = idx;
		}
		//printf("%d,%d\n", idx, *result);
	}
}




