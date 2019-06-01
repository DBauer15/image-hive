from app.Module import Module

import networkx as nx

class LinkNodes(Module):
    def __init__(self, prev_module):
        super().__init__('LinkNodes', prev_module)

    def run(self):
        super().run()
        G = nx.Graph()

