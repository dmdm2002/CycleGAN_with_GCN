import networkx as nx
import numpy as np


class Watts_Strogatz(object):
    def __init__(self, n, k, p, s=None):
        super(Watts_Strogatz, self).__init__()
        ws = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=s)
        self.adj_matrix = nx.adj_matrix(ws)

    def __call__(self):
        return self.adj_matrix.toarray().astype(np.float32)