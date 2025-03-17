import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import numpy as np


if __name__ == '__main__':
    os.chdir('../data')
    csv_file_path = 'global_relation.csv'
    df = pd.read_csv(csv_file_path)
    G = nx.from_pandas_edgelist(df, 'TG', 'TF')
    is_connected = nx.is_connected(G)
    if not is_connected:
        num_connected_components = nx.number_connected_components(G)
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len)
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = Counter(degrees)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(degree_distribution.keys(), degree_distribution.values())
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Degree')
    # plt.ylabel('Number of Nodes')
    # plt.title('Node Degree Distribution')
    # plt.show()
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    sorted_data = dict(sorted(degree_distribution.items()))
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    cumulative_values = np.cumsum(values)
    total = cumulative_values[-1]
    cumulative_probabilities = cumulative_values / total
    # plt.figure(figsize=(14, 8))
    plt.plot(labels, cumulative_probabilities, label='Original network')
    plt.xlabel('Degree', fontsize=18)
    plt.ylabel('Cumulative Probability', fontsize=18)
    plt.title('Cumulative distribution of Node Degree', fontsize=20)
    plt.grid(True)
    plt.xscale('log')

    os.chdir('..')
    csv_file_path = 'top_1_TFs_for_TGs.csv'
    df = pd.read_csv(csv_file_path)
    G = nx.from_pandas_edgelist(df, 'TG', 'TF')
    is_connected = nx.is_connected(G)
    if not is_connected:
        num_connected_components = nx.number_connected_components(G)
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len)
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = Counter(degrees)
    sorted_data = dict(sorted(degree_distribution.items()))
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    cumulative_values = np.cumsum(values)
    total = cumulative_values[-1]
    cumulative_probabilities = cumulative_values / total
    plt.plot(labels, cumulative_probabilities,  label='TOP 1 Key TF network')

    csv_file_path = 'top_3_TFs_for_TGs.csv'
    df = pd.read_csv(csv_file_path)
    G = nx.from_pandas_edgelist(df, 'TG', 'TF')
    is_connected = nx.is_connected(G)
    if not is_connected:
        num_connected_components = nx.number_connected_components(G)
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len)
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = Counter(degrees)
    sorted_data = dict(sorted(degree_distribution.items()))
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    cumulative_values = np.cumsum(values)
    total = cumulative_values[-1]
    cumulative_probabilities = cumulative_values / total
    plt.plot(labels, cumulative_probabilities,  label='TOP 3 Key TF network')

    csv_file_path = 'top_5_TFs_for_TGs.csv'
    df = pd.read_csv(csv_file_path)
    G = nx.from_pandas_edgelist(df, 'TG', 'TF')
    is_connected = nx.is_connected(G)
    if not is_connected:
        num_connected_components = nx.number_connected_components(G)
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len)
    degrees = [degree for node, degree in G.degree()]
    degree_distribution = Counter(degrees)
    sorted_data = dict(sorted(degree_distribution.items()))
    labels = list(sorted_data.keys())s
    values = list(sorted_data.values())
    cumulative_values = np.cumsum(values)
    total = cumulative_values[-1]
    cumulative_probabilities = cumulative_values / total
    plt.plot(labels, cumulative_probabilities,  label='TOP 5 Key TF network')

    plt.xscale('log')
    plt.legend()
    ax = plt.gca()
    ax.tick_params(direction='in')
    plt.savefig('Network analyze.svg', bbox_inches='tight')

