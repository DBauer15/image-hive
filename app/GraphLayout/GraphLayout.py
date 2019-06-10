from app.Module import Module
from app.GraphLayout.ImageSaliency import ImageSaliency
from app.GraphLayout.BubbleLayout import BubbleLayout
from app.GraphLayout.LayoutComposition import LayoutComposition


class GraphLayout(Module):
    def __init__(self, prev_module):
        super().__init__('2_GRAPH_LAYOUT', prev_module)

    def run(self):
        image_saliency = ImageSaliency(self._prev_model)
        bubble_layout = BubbleLayout(image_saliency, delta=0.0)
        layout_composition = LayoutComposition(bubble_layout, delta=0, out_size=700)

        final = layout_composition

        final.run()
        final.visualize()
        self._result = final.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
