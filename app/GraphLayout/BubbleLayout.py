from app.Module import Module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import cv2


class BubbleLayout(Module):
    def __init__(self, prev_module, delta=0.05):
        super().__init__('BubbleLayout', prev_module)
        self._delta = delta

    def run(self):
        super().run()

        self._result = {
            'cells': []
        }

        for i in range(len(self._data['clusters'])):
            # Extract voronoi data and create bounding polygon for the current voronoi region
            voronoi = self._data['voronoi']
            bounding_poly = self.__get_bounding_poly(voronoi.filtered_regions[i], voronoi.vertices)

            # Get the associated salient radii and assign random initial coordinates within the bounding poly
            extents = bounding_poly.get_extents()
            extx = (extents.xmax - extents.xmin)
            exty = (extents.ymax - extents.ymin)
            xmid = (extents.xmin + extents.xmax) / 2
            ymid = (extents.ymin + extents.ymax) / 2

            radii = np.array([s['r'] for s in self._data['clusters'][i]['saliencies']], dtype=np.float)
            local_centers = np.array([s['c'] for s in self._data['clusters'][i]['saliencies']], dtype=np.float)
            scale = 1 * (len(self._data['clusters'])/3) * ((extx*exty)/(len(radii)*np.mean(radii+self._delta)))
            #scale = self.__get_scale(bounding_poly, radii)
            radii *= scale
            local_centers *= scale


            # Assign random starting coordinates and insert radii
            c = 2*np.random.sample((len(radii), 5)) - 1
            c[:, 0] = xmid + c[:, 0]*(extx/10)
            c[:, 1] = ymid + c[:, 1]*(exty/10)
            c[:, 2] = radii

            # Simulate bubble movement
            c = self.__simulate_force_brownian(c, bounding_poly, 200, True)

            # Insert image origins
            c[:, 3:5] = c[:, 0:2] - local_centers

            # Create new entry for the result
            self._result['cells'].append({
                'images': self._data['clusters'][i]['images'],
                'bounding_poly': bounding_poly,
                'coordinates': c,
                'scale': scale
            })

    def __get_scale(self, bounding_poly, radii):
        mean_circ_area = np.mean(radii*radii*np.pi)

        extents = bounding_poly.get_extents()
        bounding_area = (extents.xmax - extents.xmin) * (extents.ymax - extents.ymin)
        scale = (bounding_area) / (mean_circ_area/len(radii))

        return scale

    def __get_bounding_poly(self, region, vertices):
        vertex_list = []
        for i in region:
            vertex_list.append(vertices[i])
        vertex_list.append(vertices[region[0]])

        return path.Path(vertex_list, closed=True)

    def __get_bounding_forces(self, c, p):
        f = [0.0, 0.0]
        for i in range(-1, 2):
            for j in range(-1, 2):
                point = np.copy(c[:2])
                point[0] += i * (c[2] + self._delta)
                point[1] += j * (c[2] + self._delta)
                if not p.contains_point(point):
                    f[0] += (c[0] - point[0]) * 0.15
                    f[1] += (c[1] - point[1]) * 0.15
        return f

    def __get_forces(self, c, bounding_poly, brownian):
        f = []

        for i in range(len(c)):
            x0 = 0.0
            y0 = 0.0
            if brownian:
                x0 = (np.random.sample() * 2 - 1) * 0.01
                y0 = (np.random.sample() * 2 - 1) * 0.01
            f.append([x0, y0])
            f_bounding = self.__get_bounding_forces(c[i, :], bounding_poly)
            f[i][0] += f_bounding[0]
            f[i][1] += f_bounding[1]
            for j in range(len(c)):
                if i == j:
                    continue

                dist = np.linalg.norm(c[i, :2] - c[j, :2])
                rad = (c[i, 2] + self._delta) + (c[j, 2] + self._delta)
                if dist < rad:
                    f[i][0] = f[i][0] + (c[i, 0] - c[j, 0]) * 0.15
                    f[i][1] = f[i][1] + (c[i, 1] - c[j, 1]) * 0.15
                else:
                    f[i][0] = f[i][0] + (c[i, 0] - c[j, 0]) * 0.005
                    f[i][1] = f[i][1] + (c[i, 1] - c[j, 1]) * 0.005
        return f

    def __apply_forces(self, c, f):
        for i in range(len(c)):
            c[i, 0] += f[i][0]
            c[i, 1] += f[i][1]
        return c

    def __draw(self, c, p):
        fig, ax = plt.subplots()
        ax.add_patch(patches.PathPatch(p, fill=False))
        for circ in c:
            ax.add_artist(plt.Circle((circ[0], circ[1]), circ[2], fill=False))
            ax.add_artist(plt.Circle((circ[0], circ[1]), circ[2]+self._delta, linestyle='--', fill=False))
        plt.show()

    def __simulate_force_brownian(self, c, bounding_poly, max_its, brownian):
        f = self.__get_forces(c, bounding_poly, brownian)
        i = 0

        ret = np.copy(c)

        while np.sum(np.abs(f)) > 0 and i < max_its:
            ret = self.__apply_forces(ret, f)
            f = self.__get_forces(ret, bounding_poly, brownian)
            i += 1

        return ret

    def visualize(self):
        result = self.get_module_results()

        fig, ax = plt.subplots()
        for cell in result['cells']:
            ax.add_patch(patches.PathPatch(cell['bounding_poly'], fill=False))
            for coord in cell['coordinates']:
                ax.add_artist(plt.Circle((coord[0], coord[1]), coord[2], fill=False))
                ax.add_artist(plt.Circle((coord[0], coord[1]), coord[2] + self._delta, linestyle='--', fill=False))
        plt.show()
