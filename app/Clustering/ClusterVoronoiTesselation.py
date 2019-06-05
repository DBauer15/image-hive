from app.Module import Module
import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class ClusterVoronoiTesselation(Module):
    def __init__(self, prev_module):
        super().__init__('ClusterVoronoiTesslation', prev_module)

    def run(self):
        super().run()
        points = np.concatenate((np.array([self._data['cx']]).T, np.array([self._data['cy']]).T), axis=1)
        voronoi = Voronoi(points)
        self._create_result(voronoi)

    def _create_result(self, voronoi):
        self._result = {
            'clusters': [],
            'voronoi': voronoi
        }

        num_unique_labels = len(np.unique(self._data['labels']))
        points = np.concatenate((np.array([self._data['x']]).T, np.array([self._data['y']]).T), axis=1)
        centers = np.concatenate((np.array([self._data['cx']]).T, np.array([self._data['cy']]).T), axis=1)
        for label in range(num_unique_labels):
            p = [p for i, p in enumerate(points) if self._data['labels'][i] == label]
            imgs = [img for i, img in enumerate(self._data['images']) if self._data['labels'][i] == label]
            self._result['clusters'].append({
                'images': imgs,
                'label': label,
                'points': p,
                'center': centers[label],
            })

    def visualize(self):
        result = self.get_module_results()
        plt.figure()
        voronoi_plot_2d(result['voronoi'])
        #plt.scatter(result['x'], result['y'], c=result['labels'], s=5, cmap='Dark2')
        plt.savefig('graph.pdf', dpi=800)
        plt.show()
