#pragma once
#include "data_struct.cuh"
#include "build_BRTree.cuh"

__device__ void setTreeNode2Hnode_1(Node tree_node, int end, int count, unsigned int *prefix, size_t real_size,
	HItem *sub_items, HItem_Node *sub_item_nodes, int n){
	//printf("treenode info:%d %d %d\n", tree_node.index, tree_node.count, tree_node.ancestor);
	//printf("end point=%d prefix col count=%d\n", end, real_size);
	for (int i = 0; i < real_size; i++){
		unsigned int chtmp = prefix[tree_node.id*real_size + i];
		//printf("treenode info:%d %d %d\n", tree_node.index, tree_node.count, tree_node.ancestor);
		//printf("%d %d\n", tree_node.id*real_size + i, prefix[0]);
		//printf("%d\n", chtmp);
		//display_int(chtmp, tree_node); 
		int start_bit = (i * 32) > tree_node.min_bit_loc ? (i * 32) : tree_node.min_bit_loc;//ȡ���
		int end_bit = end>tree_node.max_bit_loc ? tree_node.max_bit_loc : end;//ȡС��
	
		if (chtmp != 0){
			//printf("[%d-%d]\n", start_bit, end_bit);
			for (int j = start_bit; j <= end_bit; j++){
				unsigned int mask = (unsigned int)(ONEBIT >> (j % 32));
				if ((chtmp&mask) != 0){
					sub_items[j].id = j;
					atomicAdd(&(sub_items[j].count), count);//�����ô�������count��Ҫ�����Ǹ�ֵ�ķ�ʽ
					atomicAdd(&(sub_items[j].node_count), 1);//��������Ǵ����,nodecount�ļ���Ҫ��������������������Ǵ����
					//����Ķ�����ȷ��
					//sub_items[j].count += tree_node.count;
					//sub_items[j].node_count += 1;
					//printf("test:id=%d,count=%d,node_count=%d\n",
					//	sub_items[j].id, sub_items[j].count, sub_items[j].node_count);

					int temp_index = j*n + tree_node.index;
					atomicAdd(&sub_item_nodes[temp_index].count, count);
					sub_item_nodes[temp_index].flag = tree_node.flag;
					sub_item_nodes[temp_index].index = tree_node.index;
					//printf("id=%d,itemid=%d,real_id=%d,node_index=%d,node_flag=%d,node_count=%d\n", temp_index,
					//	j,tree_node.id,
					//	sub_item_nodes[temp_index].index, 
					//	sub_item_nodes[temp_index].flag,
					//	sub_item_nodes[temp_index].count);
				}
			}
		}
	}
}

//������ͷ���ں�,����Ҫ����߳̿�����
__global__ void parallel_build_subHtable(
	int n,
	unsigned int *leaf_prefix, unsigned int *inner_prefix, size_t real_size,//ǰ׺����
	HItem item, HItem_Node *item_nodes,//Դͷ��
	Node* leaf_nodes, Node *inner_nodes,//���������
	HItem *sub_items, HItem_Node *sub_item_nodes//Ŀ��ͷ��
	){
	int j = threadIdx.x + blockDim.x*blockIdx.x;
	//int j = threadIdx.x;
	//printf("%d %d ;", j, item.node_count);
	if (j < n){//���»����ж�����
		//	//Ѱ�����е�һ��·��from left to root
		//printf("thread:%d %d\n", j, item.node_count);
		HItem_Node temp_hitem_node = item_nodes[j];
		if (temp_hitem_node.count > 0){
			Node start;
			//printf("threadid=%d,node info:%d %d %d\n", j,temp_hitem_node.index, temp_hitem_node.flag, temp_hitem_node.count);
			unsigned int *prefix;
			if (temp_hitem_node.flag){
				start = inner_nodes[temp_hitem_node.index];
			}
			else{
				start = leaf_nodes[temp_hitem_node.index];
			}
			//printf("%d-%d������Ӧ��ǰ׺=", start.flag, start.id);
			//display_int(prefix[start.id],start);
			//printf("treenode info:%d %d %d %d\n", start.index,start.flag,start.count,start.ancestor);
			//����Ҫ��������һ��
			//printf("%d\n", start.flag);
			int end = item.id - 1;//����Ǵ����ͷ��ڵ�
			int count = temp_hitem_node.count;//����ͷ���е�count��treenode�е�count��û���õ�

			if (start.flag){
				prefix = inner_prefix;
			}
			else{
				prefix = leaf_prefix;
			}
			setTreeNode2Hnode_1(start, end, count, prefix, real_size, sub_items, sub_item_nodes, n);

			while (start.ancestor >= 0)
			{
				start = inner_nodes[start.ancestor];
				end = start.max_bit_loc;
				prefix = inner_prefix;
				setTreeNode2Hnode_1(start, end, count, prefix, real_size, sub_items, sub_item_nodes, n);
			}
		}
	}
}


__device__ int intocharArray(int x, char* items){
	int t = 0;
	if (x == 0){
		items[t++] = '0';
	}
	else{
		while (x)
		{
			items[t] = x % 10 + '0';
			x = x / 10;
			t++;
		}
		for (int i = 0; i < t / 2; i++){
			int temp = items[i];
			items[i] = items[t - i];
			items[t - i] = temp;
		}
	}
	items[t++] = ',';
	return t;
}

__device__ void printResult(FreqItems kitems, int count){

	char* items = new char[10*kitems.k + 1]();
	int t = 0;
	for (int z = 0; z < kitems.k; z++){
		t+=intocharArray(kitems.items[z], items + t);
	}

	items[t] = '\0';
	printf("%s :%d\n", items, count);
	//delete items;
	//���results����ʽ
	/*for (int z = 0; z < kitems.k; z++){
		printf("%d,",kitems.items[z]);
	}
	printf(":%d\n", count);*/
}

__global__ void parelle_mine_fptree(
	size_t n, size_t m, unsigned int minsup,
	unsigned int *leaf_prefix, unsigned int *inner_prefix, size_t real_size,//ǰ׺����
	FreqItems kitems, HItem* items, HItem_Node *item_nodes,//Դͷ��
	Node* leaf_nodes, Node *inner_nodes,//���������
	FreqItems * results
	){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	//int i = threadIdx.x;
	if (i < m){
		//printf("%d %d\n", items[i].count, minsup);
		//if (i == 2){
		if (items[i].count >= minsup){
			//������bug
			HItem itemNode = items[i];
			kitems.items[kitems.k] = itemNode.id;
			kitems.k += 1;
			//saveResult(results, z, kitems);
			printResult(kitems, itemNode.count);//���Ƶ���

			HItem *sub_items = new HItem[m]();
			HItem_Node *sub_item_nodes = new HItem_Node[n*m]();

			//printf("%d\n", itemNode.node_count);
			//HItem_Node temp_hitem_node = (item_nodes + (i*n))[0];
			//printf("node info:%d %d %d\n", temp_hitem_node.index, temp_hitem_node.flag, temp_hitem_node.count);
			parallel_build_subHtable << <n / 512 + 1, 512 >> >
				(n, leaf_prefix, inner_prefix, real_size,
				itemNode, item_nodes + (i*n), leaf_nodes, inner_nodes,
				sub_items, sub_item_nodes);
			//		//������Ҫһ��ͬ��
			//__syncthreads();
			cudaDeviceSynchronize();
			//printf("test\n");
			//printf("id=%d,count=%d,node_count=%d\n", sub_items[0].id, sub_items[0].count, sub_items[0].node_count);
			//printf("flag=%d,index=%d,count=%d\n", sub_item_nodes[0].flag, sub_item_nodes[0].index, 
			//	sub_item_nodes[0].count);
			//�鿴��ͷ��
	
			//printf("-----H_%d����ͷ������------\n",i);
			//for (int z = 0; z < m; z++){
			//	display_Hitem(sub_items, sub_item_nodes+z*n,z);
			//}
			parelle_mine_fptree << <m / 512 + 1, 512 >> >
				(n, m, minsup,
				leaf_prefix, inner_prefix, real_size,
				kitems,
				sub_items, sub_item_nodes,
				leaf_nodes, inner_nodes,
				results);
			cudaDeviceSynchronize();
			delete sub_items;
			delete sub_item_nodes;
		}
	}
}




