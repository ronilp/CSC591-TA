import copy
import pandas as pd
from igraph import *
from scipy import spatial

alp = 0.5
check_done = True


def main():
    if len(sys.argv) != 2:
        print('alpha is not entered. Please, enter alpha')
        sys.exit(1)

    global alp
    alp = float(sys.argv[1])

    attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')

    gr = Graph.Read_Edgelist("data/fb_caltech_small_edgelist.txt", directed=False)

    list_attr = list(attributes)
    gr.vs[list_attr[0]] = attributes[list_attr[0]]

    for x in list_attr:
        gr.vs[x] = attributes[x]

    vertex_clus_member = []

    global check_done

    while (check_done):
        gr, Vertex_clus_mem_tmp = SAC(gr)
        vertex_clus_member.append(Vertex_clus_mem_tmp)

    last_index = len(vertex_clus_member)
    mapping = Clustering(vertex_clus_member[-1])
    reverse_mapping = []
    for z in range(last_index - 2, -1, -1):
        reverse_mapping = []
        for i in mapping:
            temp_cluster = []
            for j in i:
                for k in Clustering(vertex_clus_member[z])[j]:
                    temp_cluster.append(k)
            reverse_mapping.append(temp_cluster)
        mapping = copy.copy(reverse_mapping)

    f = open('communities.txt', 'w')

    for i in mapping:
        f.write(str(i[0]))
        for j in i[1:]:
            f.write(", " + str(j))
        f.write("\n")


def SAC(gr):
    global alp
    Vertex_clus_mem = VertexClustering(graph=gr).membership
    gm_old = gr.modularity(Vertex_clus_mem)

    for i in range(0, len(Vertex_clus_mem)):
        Vertex_clus_mem[i] = i
    temp = 0
    stop = True

    while (temp < 15 and stop):
        print (temp)
        stop = False
        temp += 1
        temp_old_gm = gr.modularity(Vertex_clus_mem)
        for i in range(0, len(Vertex_clus_mem)):
            old_comm = Vertex_clus_mem[i]
            gain_max = -999
            max_comm = -1
            for j in range(0, len(Vertex_clus_mem)):
                Vertex_clus_mem[i] = Vertex_clus_mem[j]
                graph_mod_new = gr.modularity(Vertex_clus_mem)
                gain_mod = graph_mod_new - gm_old

                if (alp < 1):
                    gain_cos = compute_community_cosine(Vertex_clus_mem, i, j, gr)
                else:
                    gain_cos = 0

                gain_total = (alp * gain_mod) + ((1 - alp) * gain_cos)

                if (gain_total > gain_max):
                    gain_max = gain_total
                    max_comm = j

            if (max_comm != -1):
                Vertex_clus_mem[i] = Vertex_clus_mem[max_comm]
            else:
                Vertex_clus_mem[i] = old_comm

        graph_mod_diff = gr.modularity(Vertex_clus_mem) - temp_old_gm

        if (graph_mod_diff > 0.01):
            stop = True

    global check_done
    if (temp == 1):
        check_done = False

    graph_clust = Clustering(Vertex_clus_mem)

    temp1 = 0
    for i in graph_clust:
        if i:
            for j in i:
                Vertex_clus_mem[j] = temp1
            temp1 += 1

    graph_clust = Clustering(Vertex_clus_mem)

    gr.contract_vertices(mapping=Vertex_clus_mem, combine_attrs="mean")
    gr.simplify()

    number_clusters = len(graph_clust)
    if (number_clusters < 17):
        check_done = False

    return gr, Vertex_clus_mem


def compute_community_cosine(Vertex_clus_mem, i, j, gr):
    a = list(gr.vs[i].attributes().values())
    community = []
    count = 0

    for x in range(0, len(Vertex_clus_mem)):
        if Vertex_clus_mem[x] == Vertex_clus_mem[j]:
            community.append(x)
            count += 1

    sim_measure = 0

    for x in community:
        b = list(gr.vs[x].attributes().values())
        sim_measure += 1 - spatial.distance.cosine(a, b)

    return sim_measure / count


if __name__ == "__main__":
    main()
