from app.Module import Module
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from sklearn.preprocessing import normalize


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

        dist = {}
        for i, cluster in enumerate(self._data['clusters']):
            saliencies = cluster['saliencies']
            for j, point1 in enumerate(cluster['points']):
                point1_name = '{}_{}'.format(cluster['label'], j)
                dist[point1_name] = {}
                for k, point2 in enumerate(cluster['points']):
                    if k == j:
                        continue
                    point2_name = '{}_{}'.format(cluster['label'], k)
                    dist[point1_name][point2_name] = (saliencies[i]['r']+saliencies[j]['r'])

        kkpos = nx.drawing.kamada_kawai_layout(G, scale=1, dist=dist)


        for i, cluster in enumerate(self._data['clusters']):
            for j, point1 in enumerate(cluster['points']):
                point_name = '{}_{}'.format(cluster['label'], j)

                # normalize

                # offset by voronoi centroid
                kkpos[point_name] = np.add(kkpos[point_name], 5*np.array(self._data['centroids'][i]))


        plt.figure()
        #voronoi_plot_2d(self._data['voronoi'])
        nx.draw(G, with_labels=True, pos=kkpos)
        #plt.xlim((0, 1))
        #plt.ylim((0, 1))
        plt.show()

        self._result = self._data
