import random
import igraph
import pandas as pd
from igraph import *
from scipy import spatial

MAX_ITERS = 15


def load_attributes(g, attribute_list_csv):
    df = pd.read_csv(attribute_list_csv)
    for col in df.columns:
        g.vs[col] = df[col]

    return g, df.columns


def similarity(a, b):
    a = list(a.attributes().values())
    b = list(b.attributes().values())
    return 1 - spatial.distance.cosine(a, b)


def QAttr(vertices, community):
    mod = 0
    for vertex in vertices:
        for v in community:
            mod += similarity(v, vertex)
    return mod/len(community)


def QNewman(graph, vertices, community):
    m = graph.ecount()
    G = 0
    D = 0
    Q = 0
    for vertex in vertices:
        for v in community:
            # use error = False
            if graph.get_eid(v.index, vertex.index, directed=False, error=False) > -1:
                G += 1

            D += v.degree()
        Q += 1. / (2 * m) * (G - vertex.degree() / (2. * m) * D)
    return Q


def phase1(g, alpha, communities=None):
    # Initlialize nodes in separate communities
    if communities is None:
        communities = []
        for node in g.vs:
            communities.append([node])

    # nodes remaining to be added
    remaining = communities

    iter = 0
    processed = 0
    visited = []
    while iter < MAX_ITERS and len(remaining) > 0 and processed < len(remaining):
        # choose a node to move
        # to check if the node was not previously visited, use the set visited
        node = None
        while node is None or node in visited:
            node = random.choice(remaining)

        max_gain = None
        max_index = 0
        for i, community in enumerate(communities):
            if i == node[0].index:
                continue

            # current gain = composite modularity gain
            curr_gain = alpha * QNewman(g, node, community) + (1 - alpha) * QAttr(node, community)

            # record the community with highest gain
            if max_gain is None or curr_gain > max_gain:
                max_gain = curr_gain
                max_index = i

        if max_gain <= 0:
            processed += 1
            # add all single node communities to remaining communities
            remaining = []
            for community in communities:
                if len(community) == 1:
                    remaining.append(community)

            # mark node as visited
            visited = [node]
            continue

        # remove node from previous community
        communities.remove(node)
        # since gain is still positive, move the node which provides the max gain
        communities[max_index] = communities[max_index] + node
        # mark node as visited
        visited.append(node)
        processed = 0
        iter += 1

    return communities


def phase2(g, alpha, communities):
    return phase1(g, alpha, communities)


def write_results(communities, alpha):
    with open("communities_" + str(alpha) + ".txt", 'a') as f:
        for community in communities:
            temp = ""
            for node in community:
                temp = temp + str(node.index) + ','
            f.write(temp + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Incorrect Usage.\npython sac1.py <alpha value>")
        exit(0)

    alpha = float(sys.argv[1])
    edge_list_file = 'data/fb_caltech_small_edgelist.txt'
    attribute_list_csv = 'data/fb_caltech_small_attrlist.csv'

    g = igraph.load(edge_list_file)
    g, attributes = load_attributes(g, attribute_list_csv)

    detected_communities = phase1(g, alpha)
    detected_communities = phase2(g, alpha, detected_communities)

    write_results(detected_communities, alpha)
