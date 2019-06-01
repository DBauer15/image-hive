from app.Module import Module
from app.Loading.ImageSource import ImageSource
from app.Loading.ImageLoader import ImageLoader


class Loading(Module):
    def __init__(self, prev_module, source_path):
        super().__init__('0_LOADING', prev_module)
        self.source_path = source_path

    def run(self):
        image_source = ImageSource(self.source_path)
        image_loader = ImageLoader(image_source)

        image_loader.run()
        self._result = image_loader.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
