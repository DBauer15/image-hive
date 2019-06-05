from app.Module import Module
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class LinkNodes(Module):
    def __init__(self, prev_module, eps):
        super().__init__('LinkNodes', prev_module)
        self._eps = eps

    def run(self):
        super().run()

        G = nx.Graph()
        pos = {}
        for i, cluster in enumerate(self._data['clusters']):
            for j, point in enumerate(cluster['points']):
                name = '{}_{}'.format(cluster['label'], j)
                G.add_node(name)
                pos[name] = point

        for i, cluster in enumerate(self._data['clusters']):
            edges = []
            for j, point1 in enumerate(cluster['points']):
                point1_name = '{}_{}'.format(cluster['label'], j)
                for k, point2 in enumerate(cluster['points']):
                    if k == j:
                        continue
                    point2_name = '{}_{}'.format(cluster['label'], k)
                    distance = np.linalg.norm(point1-point2)
                    print(distance)
                    if distance < self._eps:
                        edges.append((point1_name, point2_name, distance))
            G.add_weighted_edges_from(edges)

        plt.figure()
        kkpos = nx.drawing.kamada_kawai_layout(G, scale=1, pos=pos)
        nx.draw(G, with_labels=True, pos=kkpos)
        plt.show()

        self._result = self._data
