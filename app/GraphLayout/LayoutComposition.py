from app.Module import Module
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import cv2


class LayoutComposition(Module):
    def __init__(self, prev_module, delta, out_size):
        super().__init__('LayoutComposition', prev_module)
        self._delta = delta
        self._out_size = out_size


    def run(self):
        super().run()

        image = np.zeros((self._out_size, self._out_size, 3), dtype=np.uint8)

        num_cells = len(self._data['cells'])
        print('0 of {} cells rendered'.format(num_cells))
        for i, cell in enumerate(self._data['cells']):
            self.__process_cell(image, cell)
            print('{} of {} cells rendered'.format(i+1, num_cells))


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._result = image

    def __process_cell(self, image, cell):

        # Get and rescale images
        scale = cell['scale']
        images = cell['images']
        self.__rescale_images(images, scale)

        # Get and rescale coordinates and bounding polygon
        coordinates = cell['coordinates'] * self._out_size
        bounding_poly = cell['bounding_poly'].transformed(transforms.Affine2D().scale(self._out_size))
        extents = bounding_poly.get_extents()
        x0 = int(extents.xmin)
        x1 = int(extents.xmax)
        y0 = int(extents.ymin)
        y1 = int(extents.ymax)

        # Iterate over all pixels in the cell
        for x in range(x0, x1):
            for y in range(y0, y1):

                # Ignore pixels outside the polygon boundary
                if not bounding_poly.contains_point([x,y]):
                    continue

                # Find the closest circle
                i = self.__nearest_circle([x,y], coordinates)

                # Get the pixel color of the corresponding image
                image[y, x] = self.__get_pixel([x,y], images[i], coordinates[i])

    def __nearest_circle(self, point, coordinates):
        least = -1
        least_id = -1

        for i, coord in enumerate(coordinates):
            dist = np.linalg.norm(point - coord[0:2]) - (coord[2]+self._delta)
            if dist < least or least == -1:
                least = dist
                least_id = i

        return least_id

    def __rescale_images(self, images, scale):
        for i in range(len(images)):
            image_scale = scale * self._out_size
            images[i] = cv2.resize(images[i], None, fx=image_scale, fy=image_scale)

    def __get_pixel(self, point, image, coordinate):
        image_coordinate = np.floor(point - coordinate[3:5]).astype(np.int)
        if np.min(image_coordinate) < 0 or np.greater_equal(image_coordinate[::-1], image.shape[:2]).any():
            return (255,255,255)
        return image[image_coordinate[1], image_coordinate[0], :]

    def visualize(self):
        result = self.get_module_results()

        fig, ax = plt.subplots()
        ax.axis('off')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.imshow(result, origin='upper', extent=[0, 1, 0, 1])
        '''
        for cell in self._data['cells']:
            scale = cell['scale']
            for i, coord in enumerate(cell['coordinates']):
                #image = cell['images'][i]
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #extent = [coord[3], coord[3]+(image.shape[1]*scale), coord[4], coord[4]+(image.shape[0]*scale)]
                #plt.imshow(image, origin='upper', extent=extent, cmap='gray')
                ax.add_artist(plt.Circle((coord[0], coord[1]), coord[2], fill=False, color='red'))

            ax.add_patch(patches.PathPatch(cell['bounding_poly'], fill=False))
        '''
        plt.show()