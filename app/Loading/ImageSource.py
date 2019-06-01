from app.Module import Module
import glob
import os.path as path

class ImageSource(Module):
    def __init__(self, source_path):
        super().__init__('ImageSource', None)
        self._source_path = source_path
        self._file_paths = None

    def run(self):
        super().run()
        print('Looking for images in path {}'.format(self._source_path))
        print(path.join(self._source_path, '*.jpg'))
        self._file_paths = glob.glob(path.join(self._source_path, '*.JPEG'))
        print('Found {} images'.format(len(self._file_paths)))
        self.has_run = True

    def get_module_results(self):
        return self._file_paths
