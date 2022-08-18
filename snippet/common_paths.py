# common paths
# https://stackoverflow.com/questions/72742661/findings-common-paths-in-two-graphs-using-python-networkx

import networkx as nx
from tqdm import tqdm

# Create a couple of random preferential attachment graphs
G = nx.barabasi_albert_graph(100, 5)
H = nx.barabasi_albert_graph(100, 5)

# Convert to directed
G = G.to_directed()
H = H.to_directed()

# Get intersection
Gintersection = nx.intersection(G, H)

# Print info for each
print(nx.info(G))
print(nx.info(H))
print(nx.info(Gintersection))

src,dst=0,1

simple_paths = nx.all_simple_paths(Gintersection, src, dst,cutoff=5)

count_lst=[]
for src in tqdm(range(100)):
    for dst in range(100):
        paths = [path for path in nx.all_simple_paths(Gintersection, src, dst,cutoff=7)]
        count_lst.extend([1 for item in paths if len(item)==5])
print(len(count_lst))
