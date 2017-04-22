/*
* author：lzy
   datetime:20160916
*/
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include "data_struct.cuh"
#include "build_bitMatrix.cuh"
#include "sort_leafnode.cuh"
#include "build_BRTree.cuh"
#include "mine_BRTree.cuh"
/*
main program
TODO: improve algothrim,unit test
analysis algothrim, nsight data analysis
*/
/*
示例输入：
f c a m p
f c a b m
f b
c b p
f c a m p
<min_support_count=3
min_support=0.6
*/
int main(int argc, char** argv)
{
	if (argc < 2){
		cout << "Usage: fpgrowth_gpu <min_support>" << endl;
		return 1;
	}
    //time1:
	cout << "----start----" << endl;
	double min_support = atof(argv[1]);
	int n_trans = 0;
	vector<vector<string>> trans;
	map<string, int> C1;
	//1.输入数据
	string line;
	while (getline(cin, line)){
		istringstream is(line);
		vector<string> items;
		string item;
		while (is >> item){
			items.push_back(item);
			map<string, int>::iterator it = C1.find(item);
			if (it != C1.end())
				C1[item] += 1;
			else
				C1[item] = 1;
		}
		trans.push_back(items);
		n_trans++;
	}

	//2.产生频繁1项集并将数据集位表化
	int n_items = 0;
	map<string, int> item_indexs;
	vector<string> item_names;
	map<string, int>::iterator it = C1.begin();
	int min_support_count = n_trans*min_support;

	for (map<string, int>::iterator it = C1.begin(); it != C1.end(); ++it){
		if (it->second >= min_support_count){
			//cout << it->first << ":" << setprecision(2) << it->second << endl;//输出F1项集
			item_indexs[it->first] = n_items++;
			item_names.push_back(it->first);
		}
	}

	//3.转化为位表
	Prefix_Array* leaf_prefixs = new Prefix_Array(n_trans, n_items);
	trans2Prefix(trans, item_indexs, leaf_prefixs, n_trans);
	//cout << "叶子节点的前缀数组为：" << endl;
	//display_prefix_array(leaf_prefixs);

	//4.构造叶子节点
	Node* leaf_node = new Node[n_trans]();
	construct_leaf_node(leaf_node, n_trans, n_items);
	//cout << "初始叶子节点是：" << endl;
	//display_leaf_node(leaf_node, n_trans);
	
	//5.对叶子节点排序去重
	Node *dev_leaf_nodes;
	cudaMalloc((void **)&dev_leaf_nodes, sizeof(Node) * n_trans);
	cudaMemcpy(dev_leaf_nodes, leaf_node, sizeof(Node) * n_trans, cudaMemcpyHostToDevice);

	for (int i = 0; i < n_trans; i += 1){
		parallel_sort_leaf << <n_trans / 512 + 1, 512 >> >(n_trans, n_items, 0, leaf_prefixs->data,
			leaf_prefixs->real_size, dev_leaf_nodes);
		parallel_sort_leaf << <n_trans / 512 + 1, 512 >> >(n_trans, n_items, 1, leaf_prefixs->data,
			leaf_prefixs->real_size, dev_leaf_nodes);

	}
	int *res = 0;
	cudaMalloc((void**)&res, sizeof(int));
	parallel_find_size << <n_trans / 512 + 1, 512 >> >(res, dev_leaf_nodes, n_trans);
	cudaDeviceSynchronize();
	int new_n_trans = 0;
	cudaMemcpy(&new_n_trans, res, sizeof(int),   cudaMemcpyDeviceToHost);
	//--------------debug-----开始
	//cudaMemcpy(leaf_node, dev_leaf_nodes, sizeof(Node) * n_trans, cudaMemcpyDeviceToHost);
	//cout << "排序后的叶子节点是：" << endl;
	//display_leaf_node(leaf_node, new_n_trans);
	//---------------debug----结束 
	cudaDeviceSynchronize();
	//6.并行构建二叉基数树
	//6-1 内部节点
	Node *dev_inner_nodes;
	cudaMalloc((void **)&dev_inner_nodes, sizeof(Node) * (new_n_trans-1));
	Prefix_Array* inner_prefixs = new Prefix_Array(new_n_trans - 1, n_items);
	//6-2 头表
	HItem *dev_items;
	HItem_Node *dev_item_nodes;
	cudaMalloc((void **)&dev_items, sizeof(HItem) * n_items);
	cudaMalloc((void **)&dev_item_nodes, sizeof(HItem_Node) * n_items*new_n_trans);

	//6-3.并行创建树的内部节点
	parallel_build_tree << <(new_n_trans - 1) / 512 + 1, 512 >> >(
		new_n_trans, n_items,
		leaf_prefixs->real_size, leaf_prefixs->data, dev_leaf_nodes,
		inner_prefixs->data, dev_inner_nodes
		);
	cudaDeviceSynchronize();
	//--------------debug-----开始
	//Node *inner_node = new Node[new_n_trans-1]();
	//cudaMemcpy(leaf_node, dev_leaf_nodes, sizeof(Node) * new_n_trans, cudaMemcpyDeviceToHost);
	//cudaMemcpy(inner_node, dev_inner_nodes, sizeof(Node) * (new_n_trans-1), cudaMemcpyDeviceToHost);
	//cout << "BRTree的叶子节点和中间节点是：" << endl;
	//display_leaf_node(leaf_node, new_n_trans);
	//cout << "----------" << endl;
	//display_leaf_node(inner_node, new_n_trans - 1);
	//cout << "中间节点的前缀数组----------" << endl;
	//display_prefix_array(inner_prefixs);
	//--------------debug-----结束
	//6-3.并行创建头表
	parallel_build_HItem << <new_n_trans / 512 + 1, 512 >> >(new_n_trans, dev_leaf_nodes,
		leaf_prefixs->data, leaf_prefixs->real_size,
		dev_items, dev_item_nodes, new_n_trans);
	cudaDeviceSynchronize();

	parallel_build_HItem << <(new_n_trans - 1) / 512 + 1, 512 >> >(new_n_trans - 1, dev_inner_nodes,
		inner_prefixs->data, inner_prefixs->real_size,
		dev_items, dev_item_nodes, new_n_trans);
	cudaDeviceSynchronize();
	//--------------debug-----开始
	//HItem *items = new HItem[n_items];
	//HItem_Node *item_nodes = new HItem_Node[n_items*new_n_trans];
	//cudaMemcpy(items, dev_items, sizeof(HItem) * n_items, cudaMemcpyDeviceToHost);
	//cudaMemcpy(item_nodes, dev_item_nodes, sizeof(HItem_Node) * n_items*new_n_trans, cudaMemcpyDeviceToHost);
	//cout << "-------------" << endl;
	//display_HItem(items, n_items);
	//cout << "头表节点-------------" << endl;
	//display_HItem_Node(item_nodes, n_items, new_n_trans);
	//--------------debug-----结束

	//7.并行挖掘二叉基数树
	FreqItems nullitems;
	nullitems.k = 0;
	unsigned int res_size = pow(2, n_items);
	FreqItems * dev_results;
	cudaMalloc((void **)&dev_results, sizeof(FreqItems)* res_size);
	/*int *z=0;
	cudaMalloc((void**)&z, sizeof(int));
	cudaMemset(z, 0, 1);*/
	//cout << new_n_trans << " " << n_items << " " << min_support_count << endl;
	parelle_mine_fptree << <n_items / 512 + 1, 512 >> >
		(new_n_trans, n_items, min_support_count,
		leaf_prefixs->data, inner_prefixs->data, n_items / 32 + 1,
		nullitems, dev_items, dev_item_nodes,
		dev_leaf_nodes, dev_inner_nodes,
		dev_results);
	cudaDeviceSynchronize();
	
	//8.结果输出
	FreqItems * results = new FreqItems[res_size]() ;
	cudaMemcpy(results, dev_results, sizeof(FreqItems) *res_size, cudaMemcpyDeviceToHost);
	display_freq(results, res_size);

	cudaFree(dev_leaf_nodes);
	cudaFree(dev_inner_nodes);
	cudaFree(dev_items);
	cudaFree(dev_item_nodes);
	cudaFree(dev_results);
	delete leaf_prefixs;
	delete inner_prefixs;
	delete results;
	/*
	注意内存泄露问题TODO:添加内存泄露控制
	*/
	cout << "----end----" << endl;
	system("pause");
    return 0;
}


