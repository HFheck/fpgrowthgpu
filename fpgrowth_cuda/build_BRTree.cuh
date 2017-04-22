#pragma once
#include "data_struct.cuh"

#define ALLBIT 0xFFFFFFFF
//Longest Common Prefix
//111111<<n 
__device__ int my_min(int x, int y){
	return x < y ? x : y;
}

__device__ int normalize(int x){
	if (x > 0)
		return 1;
	else if (x == 0)
		return 0;
	else
		return -1;
}
__device__ int my_max(int x, int y){
	return x>y ? x : y;
}

__device__ int getTransCount(Node *leaf_node, int i, int j){//可以改进为并行的
	int res = 0;
	for (int t = my_min(i, j); t <= my_max(i,j); t++){
		res += leaf_node[t].count;
	}
	return res;
}

__device__ void setLCP2InnerPrefix(unsigned int *leaf_data, Node *leaf_nodes, int i, int real_size, int max_loc,
	unsigned int *inner_data){
	int t = 0;
	while (max_loc>32)
	{
		inner_data[i*real_size + t] = leaf_data[(leaf_nodes[i].id)*real_size + t];
		max_loc -= 32;
		t++;
		if (t >= real_size)
			return;
	}
	inner_data[i*real_size + t] = (ALLBIT << (32 - max_loc))&leaf_data[(leaf_nodes[i].id)*real_size + t];
}

__device__ int LCP(unsigned int *leaf_data, Node *leaf_nodes,int i, int j, int real_size, int n){
	if (i < 0 || j < 0)
		return -1;
	if (i > n - 1 || j > n - 1)
		return -1;

	unsigned int* tmp = new unsigned int[real_size]();
	int res = 0, t = 0;
	for (t = 0; t < real_size; t++){
		tmp[t] = leaf_data[(leaf_nodes[i].id)*real_size + t] ^ leaf_data[(leaf_nodes[j].id)*real_size + t];
		if (tmp[t])
			break;
		res += 32;
	}
	//printf("%d %d\n", real_size, tmp[t]);
	if (t < real_size){
		unsigned int chtmp = tmp[t];
		for (int tt = 0; tt < 32; tt++){
			unsigned int mask = (unsigned int)(ONEBIT >> tt);
			if (chtmp&mask){
				break;
			}
			res += 1;
		}
	}
	delete tmp;

	return res;
}


__global__ void parallel_build_tree(
	int n, int m,
	int real_size, unsigned int *leaf_data, Node *leaf_nodes,//叶子节点前缀和叶子树节点
	unsigned int *inner_data, Node *inner_nodes//内部节点前缀和二叉基数树
	){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i<(n - 1)){//这里应该改为n-1
	//if (i == 1){
		//debug1-打印参数
		//printf("%d,%d,%d\n", i, n, m);
		//printf("trans_count:%d\n", trans_count[i]);
		//printf("前缀大小:%d,L%d的前缀值:%d,L%d-id值:%d\n", 
		//	real_size, i,leaf_data[i], i,leaf_nodes[i].id);
		//printf("I%d的前缀值:%d,I%d-id值:%d\n",
		//	i, inner_data[i], i,inner_nodes[i].id);//gpu没有数据组的越界控制
		//1.计算方向
		//int debug1 = LCP(leaf_data, i, i + 1, real_size, n);
		//int debug2 = LCP(leaf_data, i, i - 1, real_size, n);
		//printf("线程:%d,LCP(%d,%d)=%d,LCP(%d,%d)-%d\n",
		//	i, i, i + 1, debug1, i, i - 1, debug2);
		int d = LCP(leaf_data, leaf_nodes,i, i + 1, real_size, n) -
			LCP(leaf_data, leaf_nodes,i, i - 1, real_size, n);
		d = normalize(d);
		//printf("线程:%d,方向:%d\n", i, d);
		//2.寻找上限
		int LCP_min = LCP(leaf_data, leaf_nodes,i, i - d, real_size, n);
		int upper = 2;
		while (LCP(leaf_data, leaf_nodes,i, i + upper*d, real_size, n)>LCP_min)
			upper *= 2;
		//printf("线程:%d,上限:%d\n", i, upper);
		//3.二分搜索另一端
		int step = 0;
		for (int t = upper / 2; t >= 1; t = t / 2){
			if (LCP(leaf_data, leaf_nodes,i, i + (step + t)*d, real_size, n) > LCP_min){
				step += t;
			}
		}
		int j = i + step*d;
		//printf("线程:%d,覆盖区域:%d-%d\n", i, i, j);
		//4.使用二分搜索寻找切分点
		int LCP_node = LCP(leaf_data, leaf_nodes,i, j, real_size, n);
		step = 0;
		for (int t = upper / 2; t >= 1; t = t / 2){
			if (LCP(leaf_data, leaf_nodes,i, i + (step + t)*d, real_size, n) > LCP_node){
				step += t;
			}
		}
		int split = i + step*d + my_min(d, 0);
		//printf("线程:%d,切分点:%d\n", i, split);

		//5.
		//5.1  inner_node和leaf_node的ancestor值
		//5.2  inner_node的前缀数组LCP(i,j)的数组值
		//5.3  根据LCP值可以确定inner_node_i的max_bit_loc;
		//     和split_node和split+1_node的min_bit_loc;
		//inner_node和leaf_node的min_bit_loc和max_bit_loc
		//printf("LCP:%d,i:%d,j:%d\n", LCP_node,i,j);
		inner_nodes[i].id = i;
		inner_nodes[i].index = i;
		inner_nodes[i].flag = 1;
		inner_nodes[i].max_bit_loc = LCP_node - 1;
		inner_nodes[i].count = getTransCount(leaf_nodes, i, j);
		//printf("%d,%d,%d\n", i, inner_nodes[i].max_bit_loc, inner_nodes[i].count);
		if (i == 0){
			inner_nodes[i].ancestor = -1;
		}
		if (my_min(i, j) == split){
			leaf_nodes[split].ancestor = i;
			leaf_nodes[split].min_bit_loc = LCP_node;
		}
		else{
			inner_nodes[split].ancestor = i;
			inner_nodes[split].min_bit_loc = LCP_node;
		}

		if (my_max(i, j) == split + 1){
			leaf_nodes[split + 1].ancestor = i;
			leaf_nodes[split + 1].min_bit_loc = LCP_node;
		}
		else{
			inner_nodes[split + 1].ancestor = i;
			inner_nodes[split + 1].min_bit_loc = LCP_node;
		}
		//5.2  inner_node的前缀数组LCP(i,j)的数组值
		setLCP2InnerPrefix(leaf_data, leaf_nodes,i, real_size, LCP_node, inner_data);
	}
}

__device__ void setTreeNode2Hnode(Node tree_node, int index, unsigned int *prefix, size_t real_size,
	HItem *sub_items, HItem_Node *sub_item_nodes, int n){
	for (int i = 0; i < real_size; i++){
		unsigned int chtmp = prefix[tree_node.id*real_size + i];
		int start_bit = (i * 32) > tree_node.min_bit_loc ?
			(i * 32) : tree_node.min_bit_loc;
		int end_bit = tree_node.max_bit_loc;
		//printf("%d,%d,%d\n", start_bit, end_bit, count);//这里是正确的
		//printf("%d,%d\n", tree_node.id*real_size + i,chtmp);
		if (chtmp != 0){
			for (int j = start_bit; j <= end_bit; j++){
				unsigned int mask = (unsigned int)(ONEBIT >> (j % 32));
				if ((chtmp&mask) != 0){
					sub_items[j].id = j;
					atomicAdd(&(sub_items[j].count), tree_node.count);
					atomicAdd(&(sub_items[j].node_count), 1);//这个数字是错误的,nodecount的计算要从下面来处理
					//上面的都是正确的
					//printf("%d,%d,%d\n", j,
					//	sub_items[j].count, sub_items[j].node_count);
					int temp_index = j*n + index;
					atomicAdd(&sub_item_nodes[temp_index].count, tree_node.count);
					sub_item_nodes[temp_index].flag = tree_node.flag;
					sub_item_nodes[temp_index].index = index;
				}
			}
		}
	}
}


__global__ void parallel_build_HItem(
	int n, Node* treeNode,
	unsigned int *prefix, size_t real_size,
	HItem *items, HItem_Node *item_nodes, int uniqueTrans_count
	){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < n){
		//printf("%d,%d,%d\n", i, treeNode[i].max_bit_loc, treeNode[i].count);
		setTreeNode2Hnode(treeNode[i], i,
			prefix, real_size,
			items, item_nodes, uniqueTrans_count);
	}

}



