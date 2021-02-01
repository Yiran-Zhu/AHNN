import numpy as np

def generate_G_part(variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.zeros((25, 10))
    H[0][0], H[1][0], H[20][0] = 1, 1, 1
    H[2][1], H[3][1], H[20][1] = 1, 1, 1
    H[9][2], H[10][2], H[11][2], H[23][2], H[24][2] = 1, 1, 1, 1, 1
    H[8][3], H[9][3], H[20][3] = 1, 1, 1
    H[4][4], H[5][4], H[20][4] = 1, 1, 1
    H[5][5], H[6][5], H[7][5], H[21][5], H[22][5] = 1, 1, 1, 1, 1
    H[17][6], H[18][6], H[19][6] = 1, 1, 1
    H[0][7], H[16][7], H[17][7] = 1, 1, 1
    H[0][8], H[12][8], H[13][8] = 1, 1, 1
    H[13][9], H[14][9], H[15][9] = 1, 1, 1

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def generate_G_body(variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.zeros((25, 5))
    H[0][0], H[1][0], H[2][0], H[3][0], H[20][0], H[4][0], H[8][0], H[12][0], H[16][0] = 1, 1, 1, 1, 1, 1, 1, 1, 1
    H[8][1], H[9][1], H[10][1], H[11][1], H[23][1], H[24][1] = 1, 1, 1, 1, 1, 1
    H[4][2], H[5][2], H[6][2], H[7][2], H[21][2], H[22][2] = 1, 1, 1, 1, 1, 1
    H[16][3], H[17][3], H[18][3], H[19][3] = 1, 1, 1, 1
    H[12][4], H[13][4], H[14][4], H[15][4] = 1, 1, 1, 1

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
        

def generate_G(variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.zeros((25, 9))
    H[0][0], H[1][0], H[2][0], H[3][0], H[20][0] = 1, 1, 1, 1, 1
    H[8][1], H[9][1], H[10][1], H[11][1], H[23][1], H[24][1] = 1, 1, 1, 1, 1, 1
    H[4][2], H[5][2], H[6][2], H[7][2], H[21][2], H[22][2] = 1, 1, 1, 1, 1, 1
    H[16][3], H[17][3], H[18][3], H[19][3] = 1, 1, 1, 1
    H[12][4], H[13][4], H[14][4], H[15][4] = 1, 1, 1, 1
    H[8][5], H[9][5], H[20][5] = 1, 1, 1
    H[4][6], H[5][6], H[20][6] = 1, 1, 1
    H[0][7], H[16][7], H[17][7] = 1, 1, 1
    H[0][8], H[12][8], H[13][8] = 1, 1, 1

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # os.environ['DISPLAY'] = 'localhost:11.0'
    G = generate_G_part()
    G = G
    plt.imshow(G, cmap='gray')
    plt.show()
    print(G)