from app.Module import Module
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ImageFeaturesHistogram(Module):
    """Extracts color histogram features from given images.

    This class uses the images delivered by previous modules to extract color histograms.
    """
    def __init__(self, prev_module):
        super().__init__('ImageFeatures', prev_module)

    def run(self):
        super().run()

        self._result = {
            'images': self._data,
            'features': [],
        }

        descriptors = []
        for image in self._data:
            image = cv2.resize(image, (127, 127))
            h1 = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            h2 = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
            h3 = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(256):
                sum1 += i*h1[i]
                sum2 += i*h2[i]
                sum3 += i*h3[i]
            ssum = sum1 + sum2 + sum3
            self._result['features'].append([sum1/ssum, sum2/ssum, sum3/ssum])

    def visualize(self):
        result = self.get_module_results()
        images = result['images']
        i = np.random.randint(0, len(images))

        r, g, b = cv2.split(images[i])
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

        # plotting
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()
