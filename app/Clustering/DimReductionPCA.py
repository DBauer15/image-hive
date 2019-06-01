from app.Module import Module
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import numpy as np

class DimReductionPCA(Module):
    def __init__(self, prev_module):
        super().__init__('DimReductionPCA', prev_module)
        self._coordinate_mapping = None

    def run(self):
        super().run()
        clusters = self._prev_model.get_module_results()
        features = clusters['features']
        centers = clusters['centers']
        features = np.concatenate((features, centers))

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        pca1 = pca_result[:len(clusters['features']), 0]
        pca2 = pca_result[:len(clusters['features']), 1]
        pcac1 = pca_result[len(clusters['features']):, 0]
        pcac2 = pca_result[len(clusters['features']):, 1]

        self._coordinate_mapping = {
            'images': clusters['images'],
            'labels': clusters['labels'],
            'cx': pcac1,
            'cy': pcac2,
            'x': pca1,
            'y': pca2
        }

    def get_module_results(self):
        return self._coordinate_mapping

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
