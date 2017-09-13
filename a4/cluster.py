import conf
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import math
import itertools
import collections
from collections import defaultdict


def drawgraph(trumpid):
    G = nx.Graph()
    try:
        pkl_file = open(conf.followers_file, 'rb')
        trumpfollowers = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open(conf.followers_friends_file, 'rb')
        trump_fol_friend = pickle.load(pkl_file)
        pkl_file.close()
        labels = {}
   
        labels[trumpid]='Trump'
        G.add_node(trumpid)
    
        for fol in  trumpfollowers:
            G.add_node(fol)
            G.add_edge(trumpid,fol)
            #print('added node -adge trump -fol')
    
         
        for li in  trump_fol_friend:
            for key,val in li.items():
                for v in val:
                    if(v not in G.nodes()):
                        G.add_node(v)
                    if(not(G.has_edge(key,v) or G.has_edge(v,key))):
                        G.add_edge(key,v)
                          
        node_colors=[]
        node_size=[]       
        for n in G.nodes():
            if n in trumpfollowers:
                node_colors.append("blue")
                node_size.append(10)
            else: 
                node_colors.append("red")
                node_size.append(3)
           
        pos = nx.spring_layout(G)
        nx.draw_networkx_labels(G,pos,labels,font_size=10,font_color='green')
        nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors,node_size=node_size,width=.10)
        nx.draw_networkx_edges(G, pos=pos,width=.10)
        plt.savefig(conf.community_entire_graph_file,  dpi=2400)
        plt.show()
    except Exception as e: 
        print(e)
        raise Exception(e)
    return G   
    



def girvan_newman(G, min_number_communities):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html
    
    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.  
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * 1  # for printing
    try:
        while len(components) < min_number_communities:
            edge_to_remove = find_best_edge(G)
            #print(indent + 'removing ' + str(edge_to_remove))
            G.remove_edge(*edge_to_remove)
            components = [c for c in nx.connected_component_subgraphs(G)]
            comp=nx.connected_component_subgraphs(G)   
        result = [c.nodes() for c in components]
    except:
        pass
    return result,components
    

    
def draw_components(G,components,trumpid):
    node_colors=[]
    node_size=[]
    labels = defaultdict(None)
    for cnt,h in enumerate(components):
        pos = nx.spring_layout(h)
        if(trumpid in h):
            labels[int(trumpid)]='Trump'
            nx.draw_networkx_labels(h,pos,labels,font_size=10,font_color='green')
        nx.draw(h,pos,node_size=10,node_color=node_colors)
        plt.savefig(conf.community_partition_file_prefix+str(cnt)+'.png',  dpi=2400)
        plt.show()
 
def find_tolal_users(components):
    
    alluser=[]
    for c in components:
        for v in c:
            alluser.append(v)
    return alluser 
    
def main():
    pkl_file = open(conf.userid_file, 'rb')
    trumpid = pickle.load(pkl_file)
    pkl_file.close()  
    locG=drawgraph(trumpid)
    res,components=girvan_newman(locG,conf.number_of_partitions_min)   
    draw_components(locG,components,trumpid)
    str1=('No of communities discovered %d'%(len(components)))
    alluser=find_tolal_users(components)
    str2=('No of users in the network %d'%(len(alluser)))
    str3=('Average number  of users per communnity %f'%(len(alluser)/len(components)))
    
    
    f = open(conf.cluster_log,'w')
    f.write(str1+'\n'+str2+'\n'+str3+'\n')
    f.close()
if __name__ == '__main__':
    main()



