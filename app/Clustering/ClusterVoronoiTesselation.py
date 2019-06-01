from app.Module import Module
import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class ClusterVoronoiTesselation(Module):
    def __init__(self, prev_module):
        super().__init__('ClusterVoronoiTesslation', prev_module)
        self._result = None

    def run(self):
        super().run()
        coordinate_mapping = self._prev_model.get_module_results()
        points = np.concatenate((np.array([coordinate_mapping['cx']]).T, np.array([coordinate_mapping['cy']]).T), axis=1)

        voronoi = Voronoi(points)
        self._result = {
            'images': coordinate_mapping['images'],
            'labels': coordinate_mapping['labels'],
            'cx': coordinate_mapping['cx'],
            'cy': coordinate_mapping['cy'],
            'x': coordinate_mapping['x'],
            'y': coordinate_mapping['y'],
            'voronoi': voronoi
        }

    def get_module_results(self):
        return self._result

    def visualize(self):
        result = self.get_module_results()
        plt.figure()
        voronoi_plot_2d(result['voronoi'])
        plt.scatter(result['x'], result['y'], c=result['labels'], s=5, cmap='Dark2')
        plt.savefig('graph.pdf', dpi=800)
        plt.show()
