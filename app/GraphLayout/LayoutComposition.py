from app.Module import Module
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import cv2


class LayoutComposition(Module):
    """Composes the final image.

    This class uses the results of layout modules to create the final render.
    To render the image a raycasting like technique is used to find the closest salient region for
    each pixel in the final image. The corresponding pixel value is then applied at that point.

    Attributes:
        _delta: Delta padding to use to compose(draw) salient regions in a cluster (float)
        _out_size: Size (in pixels) of the resulting image (int)
    """
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
        """Processes a single voronoi cell of the final render.

        Pixels for the current cell are determined. For each pixel the corresponding image is found and
        the pixel value of that image at that point is applied to the final render.

        Args:
            image: Image to apply changes to (numpy/cv2 image)
            cell: Cell object to process (object)
        """
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
        """Finds the closest circle to a arbitrary point in a single cell.

        Since saliency radii are based on the original images' size we need to scale them
        to fit the 0-1 region of the final image.
        A good measure for scale was determined empirically.

        Args:
            point: Point to search from (list of ints)
            coordinates: Coordinates of salient regions (list of lists of floats)

        Returns:
            The id of the closest circle (salient region) in coordinates
        """
        least = -1
        least_id = -1

        for i, coord in enumerate(coordinates):
            dist = np.linalg.norm(point - coord[0:2]) - (coord[2]+self._delta)
            if dist < least or least == -1:
                least = dist
                least_id = i

        return least_id

    def __rescale_images(self, images, scale):
        """Scales images to appropriate size for the final rendering.

        Args:
            images: List of images to rescale (list of numpy/cv2 images)
            scale: Scale to use - determined at layout time (float)
        """
        for i in range(len(images)):
            image_scale = scale * self._out_size
            images[i] = cv2.resize(images[i], None, fx=image_scale, fy=image_scale)

    def __get_pixel(self, point, image, coordinate):
        """Gets image pixel value at a certain coordinate.

        This method determines appropriate offsets and then gets the pixel value of an image a point.
        If point lies outside the image region, a default color is returned.

        Args:
            point: Point to sample at (list of ints)
            image: Images to sample from (numpy/cv2 image)
            coordinate: Coordinate of salient region (list of floats)

        Returns:
            The id of the closest circle (salient region) in coordinates
        """
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
        plt.show()