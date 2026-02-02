
import os, gzip, json, pickle
import networkx as nx
import narsche

# Reads conceptnet gzip file, creates graph, and saves

mod_path = '/path/to/models'
# Download conceptnet from https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
cnet_path = os.path.join(mod_path, 'conceptnet-assertions-5.7.0.csv.gz')
# initialize graph
G = nx.Graph()
# read gzip file
with gzip.open(cnet_path, 'rt', encoding='utf-8') as f:
    # roop over lines
    while True:
        line = f.readline()
        if not line:
            break # end of file
        
        ### Parse relation
        # split by tab
        cols = line.split('\t')
        # get relation
        relation = cols[1].split('/')[-1]
        # ExternalURL is a pseudo-relation; therefore skip
        if relation == 'ExternalURL':
            continue
        # columns 2 and 3 are URIs
        segmented_uris = [uri.split('/') for uri in cols[2:4]]
        # get languages
        langs = [segs[2] for segs in segmented_uris]
        # both must be English
        if not all(lang == 'en' for lang in langs):
            continue
        # get node types
        node_types = [segs[1] for segs in segmented_uris]
        # both must be concepts
        if not all(node_type == 'c' for node_type in node_types):
            continue
        # get words
        words = [segs[3] for segs in segmented_uris]
        # only examine single words (no underscore)
        if not all(('_' not in word) for word in words):
            continue
        # get strength by parsing json from final column
        strength = json.loads(cols[-1])['weight']
        
        ### store in graph
        # already in graph? If not, default to 0 strength
        prev_strength = G.get_edge_data(*words, default={'strength': 0})['strength']
        # add strengths across relation types
        new_strength = strength + prev_strength
        G.add_edge(*words, strength=new_strength)

"""
# Save graph as pickle
save_path = os.path.join(mod_path, 'conceptnet.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(G, f)
"""

# Save model
mod = narsche.NetworkModel(G)
save_path = os.path.join(mod_path, 'conceptnet.mod')
mod.save(save_path)
