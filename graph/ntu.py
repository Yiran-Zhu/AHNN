import sys

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

inward_ori_index_B = [(1, 2), (2, 21), (3, 21), (4, 3), (6, 5), (7, 6),
                     (8, 7), (10, 9), (11, 10), (12, 11),
                     (14, 13), (15, 14), (16, 15), (18, 17), (19, 18),
                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_B = [(i - 1, j - 1) for (i, j) in inward_ori_index_B]
outward_B = [(j, i) for (i, j) in inward_B]
neighbor_B = inward_B + outward_B

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.B = self.get_adjacency_matrix_B(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.neighbor_B = neighbor_B 

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_adjacency_matrix_B(self, labeling_mode=None):
        if labeling_mode is None:
            return self.B
        if labeling_mode == 'spatial':
            B = tools.get_spatial_graph(num_node, self_link, inward_B, outward_B)
        else:
            raise ValueError()
        return B


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
