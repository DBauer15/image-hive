from app.Module import Module
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

class ImageFeaturesSIFT(Module):
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
            if des is not None:
                for d in des[:]:
                    descriptors.append(d)

        descriptors = np.array(descriptors)
        bag_of_words = KMeans(n_clusters=50, random_state=0).fit(descriptors)

        # TODO Build BoW histograms
        self._result['features'] = None

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
