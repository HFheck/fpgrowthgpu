#pragma once
#include <string>
#include <map>
#include <vector>
#include "data_struct.cuh"
using namespace std;

void trans2Prefix(vector<vector<string>>& trans, map<string, int>& item_indexs,
	Prefix_Array *bit_matrix, int n_unique_trans){
	for (int i = 0; i < n_unique_trans; i++){
		vector<string> items = trans[i];
		for (int j = 0; j < items.size(); j++){
			map<string, int>::iterator it = item_indexs.find(items[j]);
			if (it != item_indexs.end())
				bit_matrix->set(i, item_indexs[items[j]], true);
		}
	}
}

void construct_leaf_node(Node *leaf_node, int n, int m){
	for (int i = 0; i < n; i++){
		leaf_node[i].id = i;
		leaf_node[i].index = i;
		leaf_node[i].flag = 0;
		leaf_node[i].count = 1;
		leaf_node[i].ancestor = -1;
		leaf_node[i].min_bit_loc = 0;
		leaf_node[i].max_bit_loc = m - 1;
	}
}




