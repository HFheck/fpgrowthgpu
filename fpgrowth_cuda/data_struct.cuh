#pragma once
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ONEBIT (unsigned int)2147483648
using namespace std;

/*前缀数组：存放叶子节点和中间节点的前缀*/
struct Prefix_Array{
	size_t rows, cols;
	size_t real_size;
	unsigned int *data = NULL;

	void set(const size_t row, const size_t col, bool value){
		unsigned int tmp;
		cudaMemcpy(&tmp, &data[row*real_size + col / 32], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (value) tmp |= (unsigned int)(ONEBIT >> (col % 32));
		else tmp -= tmp & (unsigned int)(ONEBIT >> (col % 32));
		cudaMemcpy(&data[row*real_size + col / 32], &tmp, sizeof(unsigned int), cudaMemcpyHostToDevice);
	}
	bool get(const size_t row, const size_t col){
		unsigned int tmp;
		cudaMemcpy(&tmp, &data[row*real_size + col / 32], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		unsigned int foo = tmp & (unsigned int)(ONEBIT >> (col % 32));
		return (foo != 0);
	}

	Prefix_Array(size_t x, size_t y){
		rows = x;
		cols = y;
		real_size = cols / 32 + 1;
		cudaMalloc(&data, sizeof(unsigned int)*rows*real_size);//gpu上分配内存
	}
	~Prefix_Array(){
		cudaFree(data);
	}
};

void display_prefix_array(Prefix_Array *data){
	for (int i = 0; i < data->rows; i++){
		for (int j = 0; j < data->cols; j++){
			cout << data->get(i, j) << " ";
		}
		cout << endl;
	}
}
/*
头表项
*/
struct HItem{
	unsigned int id = -1;
	unsigned int count = 0;
	unsigned int node_count = 0;
};

void display_HItem(HItem *hitem, int n){
	for (int i = 0; i < n; i++){
		cout << hitem[i].id << " " <<
			hitem[i].count << " " <<
			hitem[i].node_count << " " << endl;
	}
}

/*
头表项的链接节点
*/
struct HItem_Node{
	unsigned int index = -1;
	bool flag = 1;
	unsigned int count = 0;
};

void display_HItem_Node(HItem_Node *hitem_node, int n, int m){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			cout << " {" << hitem_node[i*m + j].index << " " <<
				hitem_node[i*m + j].flag << " " <<
				hitem_node[i*m + j].count << " " << "} ";
		}
		cout << endl;
	}
}

/*
树节点
*/
struct Node{
	unsigned int id;
	unsigned int index;
	unsigned int count;
	int ancestor;
	bool flag;
	int min_bit_loc;
	int max_bit_loc;
};

void display_leaf_node(Node *leaf_node, int n){
	for (int i = 0; i < n; i++){
		cout << leaf_node[i].id << " " <<
			leaf_node[i].index << " " <<
			leaf_node[i].flag << " " <<
			leaf_node[i].count << " " <<
			leaf_node[i].ancestor << " " <<
			leaf_node[i].min_bit_loc << " " <<
			leaf_node[i].max_bit_loc << endl;
	}
}

/*
频繁k项集
*/
struct FreqItems{
	unsigned int items[30];
	unsigned int k;
	unsigned int count;
};

void display_freq(FreqItems *res, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < res[i].k; j++){
			cout << res[i].items[j] << " ";
		}
		
		cout << ":"<<res[i].count<<endl;
	}
}