from src.app.Module import Module
import cv2
import numpy as np


class ImageSaliency(Module):
    """Calculate salient regions for images.

    This class uses the results of previous modules to calculate salient regions in each image.
    The approach is rather pragmatic:
        - Calculate saliency map
        - Threshold at a high value
        - Dilate the result
        - Fit a circle around the biggest contour
        - Set circle center and radius as salient region
    """
    def __init__(self, prev_module):
        super().__init__('ImageSaliency', prev_module)

    def run(self):
        super().run()
        clusters = self._data['clusters']

        self._result = self._data

        for i, cluster in enumerate(clusters):
            self._result['clusters'][i]['saliencies'] = []
            saliencies = []
            for img in cluster['images']:
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                (success, saliency_map) = saliency.computeSaliency(img)
                saliency_map = (saliency_map * 255).astype("uint8")
                ret, thresh = cv2.threshold(saliency_map, 200, 255, 0)
                kernel = np.ones((5, 5), dtype=np.uint8)
                thresh = cv2.dilate(thresh, kernel=kernel, iterations=3)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(contour)
                saliencies.append({
                    'r': int(radius),
                    'c': (int(x), int(y))
                })
            self._result['clusters'][i]['saliencies'] = saliencies
