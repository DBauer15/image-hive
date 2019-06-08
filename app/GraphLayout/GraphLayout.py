from app.Module import Module
from app.GraphLayout.ImageSaliency import ImageSaliency
from app.GraphLayout.LinkNodes import LinkNodes
from app.GraphLayout.ConstraintLayout import ConstraintLayout


class GraphLayout(Module):
    def __init__(self, prev_module):
        super().__init__('2_GRAPH_LAYOUT', prev_module)

    def run(self):
        image_saliency = ImageSaliency(self._prev_model)
        link_nodes = LinkNodes(image_saliency, 1)

        link_nodes.run()
        link_nodes.visualize()
        self._result = link_nodes.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
