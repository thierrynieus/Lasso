
def deg_stat(params_neurons):

    # params_neurons['sigma']=0.1
    params_netw1 = nu.create_conn_mat_spatial(params_neurons)
    g = nx.DiGraph(params_netw1['conn_mat'])
    x = g.in_degree
    degrees = [val for (node, val) in x]
    m1, s1 = np.mean(degrees), np.std(degrees)
    print(m1, s1, s1/m1)
    params_netw2 = nu.create_conn_mat(params_neurons)
    g = nx.DiGraph(params_netw2['conn_mat'])
    x = g.in_degree
    degrees = [val for (node, val) in x]
    m2, s2 = np.mean(degrees), np.std(degrees)
    print(m2, s2, s2/m2)
    return params_netw1, params_netw2, s1/m1
