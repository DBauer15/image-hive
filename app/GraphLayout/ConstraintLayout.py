from app.Module import Module
import numpy as np
import app.tools.laguerre_voronoi_2d as lv2

class ConstraintLayout(Module):
    def __init__(self, prev_module):
        super().__init__('ConstraintLayout', prev_module)

    def run(self):
        super().run()

        self._result = self._data

        for i, cluster in enumerate(self._data['clusters']):
            points = np.asarray([s['c'] for s in cluster['saliencies']])
            radii = np.asarray([s['r'] for s in cluster['saliencies']])
            tri_list, V = lv2.get_power_triangulation(points, radii)
            voronoi_cell_map = lv2.get_voronoi_cells(points, V, tri_list)
            lv2.display(points, radii, tri_list, voronoi_cell_map)
