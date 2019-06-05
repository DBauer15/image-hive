from app.Module import Module
from app.tools.lloyd import Lloyd
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
        lloyd = Lloyd(points=voronoi.points)
        lloyd.generate_voronoi()
        lloyd.vor

        self._result = self._data
        self._result['voronoi'] = lloyd.relax_points(2)

    def visualize(self):
        result = self.get_module_results()
        plt.figure()
        voronoi_plot_2d(result['voronoi'])
        plt.savefig('graph.pdf', dpi=800)
        plt.show()
