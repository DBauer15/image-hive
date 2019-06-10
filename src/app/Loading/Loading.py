from src.app.Module import Module
from src.app.Loading.ImageSource import ImageSource
from src.app.Loading.ImageLoader import ImageLoader


class Loading(Module):
    """Container class for all image loading modules.

    This class accumulates all image loading modules and executes them with defined settings.

    Attributes:
        source_path: Path to the image collection (string)
        source_file_type: File format to look for (string)
        resize_width: Width to use to resize bigger input images (int)
    """
    def __init__(self, prev_module, source_path='images', source_file_type='jpeg', resize_width=127):
        super().__init__('0_LOADING', prev_module)
        self.source_path = source_path
        self.source_file_type = source_file_type
        self.resize_width = resize_width

    def run(self):
        """Runs all layout sub-modules.
        """
        image_source = ImageSource(source_path=self.source_path, extension=self.source_file_type)
        image_loader = ImageLoader(image_source, width=self.resize_width)

        image_loader.run()
        self._result = image_loader.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
