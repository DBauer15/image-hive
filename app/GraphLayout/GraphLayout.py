from app.Module import Module
from app.GraphLayout.ImageSaliency import ImageSaliency
from app.GraphLayout.BubbleLayout import BubbleLayout
from app.GraphLayout.LayoutComposition import LayoutComposition


class GraphLayout(Module):
    """Container class for all layout modules.

    This class accumulates all layout modules and executes them with defined settings.

    Attributes:
        layout_delta: Delta padding to use to arrange salient regions in a cluster (float)
        composition_delta: Delta padding to use to compose(draw) salient regions in a cluster (float)
        composition_size: Size (in pixels) of the resulting image (int)
    """
    def __init__(self, prev_module, layout_delta=0.0, composition_delta=0.0, composition_size=700):
        super().__init__('2_GRAPH_LAYOUT', prev_module)
        self.layout_delta = layout_delta
        self.composition_delta = composition_delta
        self.composition_size = composition_size

    def run(self):
        """Runs all layout sub-modules.
        """
        image_saliency = ImageSaliency(self._prev_model)
        bubble_layout = BubbleLayout(image_saliency, delta=self.layout_delta)
        layout_composition = LayoutComposition(bubble_layout, delta=self.composition_delta, out_size=self.composition_size)

        final = layout_composition

        final.run()
        final.visualize()
        self._result = final.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
