import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter

class CommunityDetector:
    """
    Detects communities in migration networks using various algorithms.
    Compares community structure with bottleneck locations.
    """
    
    def __init__(self):
        self.community_results = {}
    
    def detect_communities(self, network, method='louvain'):
        """
        Detect communities in the network.
        
        Args:
            network: NetworkX graph
            method: 'louvain', 'leiden', 'greedy_modularity', or 'label_propagation'
            
        Returns:
            dict: {
                'communities': {node_id: community_id},
                'modularity': float,
                'num_communities': int
            }
        """
        if network.number_of_nodes() == 0:
            return {'communities': {}, 'modularity': 0, 'num_communities': 0}
        
        # Convert to undirected for community detection
        undirected = network.to_undirected()
        
        if method == 'louvain':
            try:
                import community.community_louvain as community_louvain
                communities = community_louvain.best_partition(undirected)
                modularity = community_louvain.modularity(communities, undirected)
            except ImportError:
                try:
                    # Try alternative import name
                    import networkx.algorithms.community as nx_comm
                    communities_generator = nx_comm.louvain_communities(undirected, seed=42)
                    communities = {}
                    for i, comm in enumerate(communities_generator):
                        for node in comm:
                            communities[node] = i
                    modularity = nx_comm.modularity(undirected, communities_generator)
                except (ImportError, AttributeError):
                    print("python-louvain not installed. Using greedy_modularity instead.")
                    print("Install with: pip install python-louvain")
                    return self.detect_communities(network, method='greedy_modularity')
        
        elif method == 'greedy_modularity':
            communities_generator = nx.community.greedy_modularity_communities(undirected)
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
            modularity = nx.community.modularity(undirected, communities_generator)
        
        elif method == 'label_propagation':
            communities_generator = nx.community.asyn_lpa_communities(undirected)
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
            # Calculate modularity
            partition = [set(comm) for comm in communities_generator]
            modularity = nx.community.modularity(undirected, partition)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'communities': communities,
            'modularity': modularity,
            'num_communities': len(set(communities.values()))
        }
    
    def compare_with_bottlenecks(self, network, communities, bottleneck_locations):
        """
        Compare community structure with bottleneck locations.
        
        Returns:
            dict: Analysis of overlap between communities and bottlenecks
        """
        bottleneck_node_ids = {bn['node_id'] for bn in bottleneck_locations['node_bottlenecks']}
        
        # Count bottlenecks per community
        bottlenecks_per_community = Counter()
        nodes_per_community = Counter()
        
        for node_id, comm_id in communities.items():
            nodes_per_community[comm_id] += 1
            if node_id in bottleneck_node_ids:
                bottlenecks_per_community[comm_id] += 1
        
        # Calculate statistics
        analysis = {
            'total_bottlenecks': len(bottleneck_node_ids),
            'total_communities': len(set(communities.values())),
            'bottlenecks_per_community': dict(bottlenecks_per_community),
            'nodes_per_community': dict(nodes_per_community),
            'bottleneck_density': {}
        }
        
        # Calculate bottleneck density per community
        for comm_id in set(communities.values()):
            bottleneck_count = bottlenecks_per_community.get(comm_id, 0)
            node_count = nodes_per_community.get(comm_id, 0)
            analysis['bottleneck_density'][comm_id] = (
                bottleneck_count / node_count if node_count > 0 else 0
            )
        
        # Find communities with highest bottleneck density
        sorted_communities = sorted(
            analysis['bottleneck_density'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        analysis['high_bottleneck_communities'] = [
            {'community_id': comm_id, 'density': density}
            for comm_id, density in sorted_communities[:5]
        ]
        
        return analysis
    
    def get_community_locations(self, network, communities):
        """
        Get geographic locations of communities.
        
        Returns:
            pd.DataFrame: Community locations with centroids
        """
        community_data = []
        
        for comm_id in set(communities.values()):
            comm_nodes = [n for n, c in communities.items() if c == comm_id]
            
            if not comm_nodes:
                continue
            
            # Calculate centroid
            lats = [network.nodes[n].get('lat', 0) for n in comm_nodes if network.nodes[n].get('lat')]
            lons = [network.nodes[n].get('lon', 0) for n in comm_nodes if network.nodes[n].get('lon')]
            
            if lats and lons:
                community_data.append({
                    'community_id': comm_id,
                    'centroid_lat': np.mean(lats),
                    'centroid_lon': np.mean(lons),
                    'num_nodes': len(comm_nodes),
                    'total_cranes': sum(network.nodes[n].get('total_cranes', 0) for n in comm_nodes)
                })
        
        return pd.DataFrame(community_data)

