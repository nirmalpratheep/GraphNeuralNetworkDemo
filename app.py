from flask import Flask, render_template, request, jsonify
import networkx as nx
import numpy as np
import json

app = Flask(__name__)

# Default graph
def create_default_graph(num_nodes=6, num_edges=10):
    # Validate inputs
    num_nodes = max(1, min(10, num_nodes))  # Clamp to 1-10 nodes
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2  # Max edges for undirected graph
    num_edges = max(1, min(20, min(max_possible_edges, num_edges)))  # Clamp to 1-20 and possible edges
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = set()
    
    # First ensure connectivity with minimum spanning tree (n-1 edges)
    nodes = list(range(num_nodes))
    for i in range(num_nodes-1):
        edges.add(tuple(sorted((nodes[i], nodes[i+1]))))
    
    # Add remaining random edges up to target
    while len(edges) < num_edges:
        a = np.random.randint(0, num_nodes)
        b = np.random.randint(0, num_nodes)
        if a != b:
            edges.add(tuple(sorted((a, b))))
    
    G.add_edges_from(list(edges))
    return G

# Simple message-passing layers to illustrate behavior
def apply_layer(G, embeddings, layer_type, pooling):
    """
    embeddings: (n,) numpy array of floats for simplicity
    returns new embeddings and a record of operations for visualization
    """
    n = len(embeddings)
    new_emb = np.zeros_like(embeddings)
    ops = []

    for node in G.nodes():
        neigh = list(G.neighbors(node))
        neigh_vals = embeddings[neigh] if len(neigh) > 0 else np.array([])
        if pooling == 'mean':
            pooled = np.mean(neigh_vals) if len(neigh_vals) > 0 else 0.0
        elif pooling == 'max':
            pooled = np.max(neigh_vals) if len(neigh_vals) > 0 else 0.0
        else:  # attention (simple softmax attention using node and neighbor values)
            if len(neigh_vals) == 0:
                pooled = 0.0
            else:
                scores = np.exp(neigh_vals - np.max(neigh_vals))
                weights = scores / np.sum(scores)
                pooled = np.sum(weights * neigh_vals)

        # layer types influence how node combines its own value and pooled neighbor info
        if layer_type == 'GCN':
            # simple average of self and pooled
            updated = 0.5 * embeddings[node] + 0.5 * pooled
        elif layer_type == 'GIN':
            # emphasize self more (sum-like)
            updated = embeddings[node] * 1.2 + pooled * 0.8
        elif layer_type == 'GraphSage':
            # concatenate-style mimic: average then non-linear (tanh)
            updated = np.tanh(embeddings[node] + pooled)
        elif layer_type == 'GAN':
            # not a real GNN, but for demo: introduce a gating behavior
            gate = 1.0 / (1.0 + np.exp(-embeddings[node]))
            updated = gate * embeddings[node] + (1 - gate) * pooled
        else:
            updated = embeddings[node]

        new_emb[node] = float(updated)
        ops.append({
            'node': int(node),
            'self': float(embeddings[node]),
            'pooled': float(pooled),
            'updated': float(updated),
            'neighbors': [int(x) for x in neigh]
        })

    return new_emb, ops

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.json or {}
    layer_type = data.get('model', 'GCN')
    num_layers = int(data.get('layers', 2))
    pooling = data.get('pooling', 'mean')
    num_nodes = int(data.get('nodes', 6))
    num_edges = int(data.get('edges', 10))
    regenerate = bool(data.get('regenerate', False))
    
    # Use static seed if not regenerating
    if not regenerate:
        np.random.seed(42)
    
    G = create_default_graph(num_nodes, num_edges)
    
    # Generate initial embeddings - constant unless regenerating
    if not regenerate:
        np.random.seed(42)  # Use same seed for consistent embeddings
    emb = 0.5 + 1.5 * np.random.random(num_nodes)
    
    timeline = []
    timeline.append({'layer': 0, 'embeddings': emb.tolist(), 'ops': []})

    for L in range(1, num_layers + 1):
        emb, ops = apply_layer(G, emb, layer_type, pooling)
        timeline.append({'layer': L, 'embeddings': emb.tolist(), 'ops': ops})

    # send graph structure and timeline
    graph_data = {
        'nodes': [{'id': i} for i in G.nodes()],
        'links': [{'source': int(u), 'target': int(v)} for u, v in G.edges()]
    }

    return jsonify({'graph': graph_data, 'timeline': timeline})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

