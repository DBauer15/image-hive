from app.Module import Module
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import numpy as np

class DimReductionPCA(Module):
    def __init__(self, prev_module):
        super().__init__('DimReductionPCA', prev_module)

    def run(self):
        super().run()
        features = self._data['features']
        centers = self._data['centers']
        features = np.concatenate((features, centers))

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)

        # Normalize results
        pca_result[:, 0] = (-np.min(pca_result[:, 0]) + pca_result[:, 0])/(-np.min(pca_result[:, 0]) + np.max(pca_result[:, 0]))
        pca_result[:, 1] = (-np.min(pca_result[:, 1]) + pca_result[:, 1])/(-np.min(pca_result[:, 1]) + np.max(pca_result[:, 1]))

        # Extract coordinates
        pca1 = pca_result[:len(self._data['features']), 0]
        pca2 = pca_result[:len(self._data['features']), 1]
        pcac1 = pca_result[len(self._data['features']):, 0]
        pcac2 = pca_result[len(self._data['features']):, 1]

        self._result = {
            'images': self._data['images'],
            'labels': self._data['labels'],
            'cx': pcac1,
            'cy': pcac2,
            'x': pca1,
            'y': pca2
        }

    def visualize(self, glyph_size=0.05):
        result = self.get_module_results()
        plt.figure()
        plt.axis([np.min(result['x']), np.max([result['x']]), np.min(result['y']), np.max(result['y'])])
        plt.scatter(result['x'], result['y'], c=result['labels'], s=0.5, cmap='Dark2')
        for i, image in enumerate(result['images']):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            extent = [result['x'][i]-(glyph_size/2), result['x'][i]+(glyph_size/2), result['y'][i]-(glyph_size/2), result['y'][i]+(glyph_size/2)]
            plt.imshow(img, origin='upper', extent=extent, cmap='gray')
        plt.savefig('graph.pdf', dpi=800)
        plt.show()
