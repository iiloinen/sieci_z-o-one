import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from pyvis.network import Network
import networkx as nx
import json
import cv2
from matplotlib.lines import Line2D

class Album():
    def __init__(self, name, sales, year):
        self.name = name
        self.sales  = sales
        self.year = year


class InitialDataLoader():
    def __init__(self):
        pass
    
    def load_data(self):
        with open('InitialData.json','r') as initial_data_file:
            initial_data_unparsed = initial_data_file.read()
        initial_data = json.loads(initial_data_unparsed)
        initial_data_file.close()

        album_sales_data = initial_data["album_sales_data"]
        distribution_avid_fans = initial_data["distribution_avid_fans"]
        generations = initial_data["generations"]
        album_pagerank = initial_data["album_pagerank"] 
        color_map = initial_data["color_map"]

        return (album_sales_data, distribution_avid_fans, generations, album_pagerank,color_map)

class PageRankCalulator():
    def __init__(self):
        pass

    def compute_pagerank(self,graph , damping_factor=0.85, max_iterations=2, tolerance=1.0e-6):
        num_nodes = len(graph.nodes)
        initial_value = 1 / num_nodes

        pagerank_scores = {node: initial_value for node in graph.nodes}

        for _ in range(max_iterations):
            new_pagerank_scores = {}


            for node in graph.nodes:
                incoming_neighbors = list(graph.predecessors(node))
                pagerank_sum = sum(pagerank_scores[neighbor] / graph.out_degree(neighbor) for neighbor in incoming_neighbors)
                new_pagerank = (1 - damping_factor) / num_nodes + damping_factor * pagerank_sum
                new_pagerank_scores[node] = new_pagerank


            convergence = all(abs(new_pagerank_scores[node] - pagerank_scores[node]) < tolerance for node in graph.nodes)


            pagerank_scores = new_pagerank_scores

            if convergence:
                break

        return pagerank_scores


class SimulationGraph():
    def __init__(self,distribution_avid_fans, color_map, album_sales_data, generations, total_nodes = 200, edges_per_new_node = 5 ):
        self.generations = generations
        self.pageRankCalculator = PageRankCalulator()
        self.distribution_avid_fans = distribution_avid_fans
        self.album_sales_data = album_sales_data
        self.color_map = color_map
        self.total_nodes = total_nodes
        self.edges_per_new_node = edges_per_new_node
        self.PRGraph = self._assemble_ba_graph()
        self.baGraph = self.PRGraph.copy()
        self.graph_pos = nx.kamada_kawai_layout(self.PRGraph)
        self.album_listeners_dict = {}


    def _assemble_ba_graph(self):
        self.undirected_original_graph = nx.Graph()
        z = nx.barabasi_albert_graph(self.total_nodes, self.edges_per_new_node)
        BA_G = nx.DiGraph()
        BA_G.add_nodes_from(z.nodes)
        BA_G.add_edges_from(z.edges)

        # demografic
        assigned_demographics = []
        for demographic, count in self.distribution_avid_fans.items():
            assigned_demographics.extend([demographic] * int(count * self.total_nodes))

        while len(assigned_demographics) < self.total_nodes:
            assigned_demographics.append(random.choice(list(self.distribution_avid_fans.keys())))
        random.shuffle(assigned_demographics)

      

        self.ba_pagerank = self.pageRankCalculator.compute_pagerank(BA_G)
        self.undirected_original_graph.add_nodes_from(BA_G.nodes)


        for i, node in enumerate(BA_G.nodes()):
            BA_G.nodes[node]['type'] = "fan group"
            BA_G.nodes[node]['demographic'] = assigned_demographics[i]
            BA_G.nodes[node]['score'] = self.ba_pagerank[node]

            self.undirected_original_graph.nodes[node]['type'] = "fan group"
            self.undirected_original_graph.nodes[node]['demographic'] = assigned_demographics[i]
            self.undirected_original_graph.nodes[node]['score'] = self.ba_pagerank[node]
        #demografic ^^^

        edges_to_validate = list(BA_G.edges()).copy()
        for (node_first, node_second) in edges_to_validate:
            if random.random() > self.generations[BA_G.nodes[node_first]['demographic']][BA_G.nodes[node_second]['demographic']]:
                BA_G.remove_edge(node_first,node_second)

        self.undirected_original_graph.add_edges_from(BA_G.edges)
            
        return BA_G
        

    def add_album_to_the_graph(self, album, connection_coeff = 100):
        number_of_edges_to_add = int(album.sales * connection_coeff)
        original_node_list = list(self.baGraph.nodes())
        new_album_properties = {'type': 'album', 'sales': album.sales}
        self.PRGraph.add_node(album.name, **new_album_properties)
        self.album_listeners_dict[album.name] = {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0}
        for edge_index in range(number_of_edges_to_add):
            new_listener_group = random.choice(original_node_list)
            self.PRGraph.add_edge(new_listener_group, album.name)
            self.album_listeners_dict[album.name][self.baGraph.nodes()[new_listener_group]['demographic']] +=1

        new_node_pos = (np.random.rand(2) -.5) *2.5

        self.graph_pos[album.name] = new_node_pos
        



    def show_graph(self, year):
        plt.figure(figsize=(10, 8))  
        network = Network()
        def scale_node_size(sales, min_size=10, max_size=30):
            min_sales, max_sales = min(self.album_sales_data.values(), key=lambda x: x['sales'])['sales'], max(self.album_sales_data.values(), key=lambda x: x['sales'])['sales']
            return min_size + (max_size - min_size) * ((sales - min_sales) / (max_sales - min_sales))

        node_sizes = []
        node_colors = []
        labels = {}

        updated_page_rank_scores = self.pageRankCalculator.compute_pagerank(self.PRGraph)
        for node in self.PRGraph.nodes():
            node_type = self.PRGraph.nodes[node].get('type')
            demographic = self.PRGraph.nodes[node].get('demographic')
            if node_type == 'album':
                node_color = self.color_map['album']
                node_size = scale_node_size(self.album_sales_data[node]['sales'])*100*updated_page_rank_scores[node]
                node_sizes.append(node_size)
                node_colors.append(node_color)
            if node_type != 'album':
                node_color = self.color_map[demographic]
                node_size = updated_page_rank_scores[node] *10000 


                node_sizes.append(node_size)
                node_colors.append(node_color)

            network.add_node(node, title=demographic, color=node_color, size=node_size)

        for edge in self.PRGraph.edges():
            network.add_edge(edge[0], edge[1])

        nx.draw_networkx_edges(self.PRGraph, pos=self.graph_pos, node_size= node_sizes, alpha=0.02, arrowstyle='-')
        nx.draw_networkx_nodes(self.PRGraph, pos=self.graph_pos, node_color=node_colors, node_size=node_sizes)

        plt.title(f'Fani albumów Taylor Swift w {year}')

        custom_handles = [
            Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='Millenials'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='Baby Boomers'),
            Line2D([0], [0], color='green',marker='o', linestyle='None', markersize=10, label='Gen Xers'),
            Line2D([0], [0], color='orange', marker='o', linestyle='None', markersize=10, label='Gen Z'),
            Line2D([0], [0], color='yellow', marker='o', linestyle='None', markersize=10, label='Album')
        ]


        plt.legend(handles=custom_handles)


        plt.savefig(f'{year}.png')
        plt.clf()

    
    def analyze_network(self, graph):
        degrees = dict(graph.degree())
        avg_degree = np.mean(list(degrees.values()))

        max_degree = max(degrees.values())
        nodes_with_max_degree = [node for node, degree in degrees.items() if degree == max_degree]
        
        clustering_coeffs = nx.clustering(graph)
        avg_clustering_coeff = np.mean(list(clustering_coeffs.values()))

        largest_cc = max(nx.connected_components(graph.to_undirected()), key=len)
        subgraph = graph.subgraph(largest_cc)

        print(f"Średni stopień węzła: {avg_degree}")
        print(f"Wierzchołek(-i) o największym stopniu: {nodes_with_max_degree}, Stopień: {max_degree}")
        print(f"Średni współczynnik klastrowania: {avg_clustering_coeff}")


def make_a_video(start_year, end_year):

    images = ['2006.png','2008.png','2010.png','2012.png','2014.png','2017.png','2019.png','2020.png','2023.png',]
    images.sort()

    frm = cv2.imread(f'2023.png')
    height, width, layers = frm.shape

    video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print("\nvideo finished!", end='')

if __name__ =="__main__":

    initialDataLoader = InitialDataLoader()

    (
    album_sales_data,
    distribution_avid_fans,
    generations,
    album_pagerank,
    color_map
    ) = initialDataLoader.load_data()
    


    simulationGraph = SimulationGraph(distribution_avid_fans, color_map, album_sales_data, generations)
    BA_G = simulationGraph.PRGraph
    simulationGraph.analyze_network(simulationGraph.baGraph)

    if True:
        for current_year in range(2005, 2024):  
            albums_dropped_this_year = [Album(key, values['sales'], values['year']) for key, values in album_sales_data.items() if int(values['year']) == current_year]

            for album in albums_dropped_this_year:
                simulationGraph.add_album_to_the_graph(album)

            if len(albums_dropped_this_year) > 0:
                simulationGraph.show_graph(current_year)
                simulationGraph.analyze_network(simulationGraph.PRGraph)
                print('\n___________________________________________________________\n')


        print('Kto słucha czego: \n')
        print(simulationGraph.album_listeners_dict)

    if True:
        original_degrees = simulationGraph.undirected_original_graph.degree()
        summaryczne_degree_dict = {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0}
        for (node, degree) in original_degrees:
            summaryczne_degree_dict[simulationGraph.undirected_original_graph.nodes[node]['demographic']] +=degree

        print("\nKto jaki jest ważny: \n")
        print(summaryczne_degree_dict)

        kto_z_kim_dikszynary ={"Millennials": {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0},
                                'Baby Boomers': {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0},
                                'Gen Xers': {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0},
                                'Gen Z': {"Millennials": 0, 'Baby Boomers': 0, 'Gen Xers': 0,'Gen Z': 0}}
        for (node_first, node_second) in simulationGraph.undirected_original_graph.edges():
            kto_z_kim_dikszynary[simulationGraph.undirected_original_graph.nodes[node_first]['demographic']][simulationGraph.undirected_original_graph.nodes[node_second]['demographic']]+=1

        print("\nkto z kim\n")

        print(kto_z_kim_dikszynary)
    if True:
        make_a_video(2005,2023)

