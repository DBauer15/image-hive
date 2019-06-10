from app.Module import Module
import cv2


class ImageLoader(Module):
    def __init__(self, prev_module, width=127):
        super().__init__('ImageLoader', prev_module)
        self._width = width

    def run(self):
        super().run()
        self._result = []
        for file_name in self._data:
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self._width, int(self._width * img.shape[0]/img.shape[1])))
            self._result.append(img)
        print('Loaded {} images'.format(len(self._result)))
        self.has_run = True
