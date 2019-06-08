from app.Module import Module
import numpy as np
import cv2
import app.tools.clipped_voronoi as clv
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class ClusterVoronoiTesselation(Module):
    def __init__(self, prev_module, num_iterations=15):
        super().__init__('ClusterVoronoiTesslation', prev_module)
        self._num_iterations = num_iterations

    def run(self):
        super().run()
        points = np.concatenate((np.array([self._data['cx']]).T, np.array([self._data['cy']]).T), axis=1)

        centroids, voronoi = clv.cvt(points, [0, 1, 0, 1], self._num_iterations)
        self._create_result(centroids, voronoi)

    def _create_result(self, centroids, voronoi):
        self._result = {
            'clusters': [],
            'voronoi': voronoi,
            'centroids': centroids
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
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()
