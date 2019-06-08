from app.Module import Module
from app.tools.lloyd import Lloyd
import app.tools.clipped_voronoi as clv
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import numpy as np

# TODO Not used now. Faulty
class CVT(Module):
    def __init__(self, prev_module, num_iterations = 6):
        super().__init__('CVT', prev_module)
        self._num_iterations = num_iterations

    def run(self):
        super().run()
        voronoi = self._data['voronoi']
        voronoi = clv.cvt(voronoi.filtered_points, [0, 1, 0, 1], self._num_iterations)

        self._result = self._data
        self._result['voronoi'] = voronoi

    def visualize(self):
        result = self.get_module_results()
        plt.figure()
        voronoi_plot_2d(result['voronoi'])
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.savefig('graph.pdf', dpi=800)
        plt.show()
