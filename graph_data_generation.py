# node_types.py
import numpy as np
import scipy.sparse as sp
import os

def define_node_types():
    """
    定义每个节点（变量）的类型。

    返回:
        node_types (np.ndarray): 长度为10的一维数组，每个元素为节点类型（0, 1, 2）。
    """
    # 示例分配，可以根据实际需求调整
    node_types = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    return node_types


def define_adjacency_matrices(graph_data_path, top_k=3):
    """
    手动定义并保存邻接矩阵。

    参数:
        graph_data_path (str): 图数据文件的保存目录路径。
        top_k (int): 不同尺度的数量。
    """
    os.makedirs(graph_data_path, exist_ok=True)

    # 示例：定义三个尺度的邻接矩阵
    for scale in range(top_k):
        # 初始化一个10x10的全零矩阵
        adj_matrix = np.zeros((10, 10), dtype=np.float32)

        if scale == 0:
            # 尺度0: 全连接图
            adj_matrix = np.ones((10, 10), dtype=np.float32) - np.eye(10, dtype=np.float32)
        elif scale == 1:
            # 尺度1: 邻近连接（链状）
            np.fill_diagonal(adj_matrix[:-1, 1:], 1)
            np.fill_diagonal(adj_matrix[1:, :-1], 1)
        elif scale == 2:
            # 尺度2: 分组连接，例如每3个节点为一组，全连接
            adj_matrix = np.array([
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float32)
        else:
            # 其他尺度可根据需求定义
            adj_matrix = np.zeros((10, 10), dtype=np.float32)

            # 转换为稀疏矩阵
        adj_sparse = sp.coo_matrix(adj_matrix)
        adj_file = os.path.join(graph_data_path, f'adj_scale{scale}.npz')
        sp.save_npz(adj_file, adj_sparse)

    print(f"成功定义并保存 {top_k} 个尺度的邻接矩阵到 {graph_data_path}。")

if __name__ == '__main__':
    node_types = define_node_types()
    np.save('./graph_data/node_types.npy', node_types)
    print("节点类型已保存到 ./graph_data/node_types.npy")

    graph_data_path = './graph_data/'  # 确保与主脚本中的路径一致
    define_adjacency_matrices(graph_data_path)