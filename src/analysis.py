import networkx as nx
import pandas as pd

class FlowAnalyzer:
    """
    Performs Max-Flow Min-Cut analysis on the migration network.
    """
    
    def __init__(self, migration_network):
        """
        Args:
            migration_network (nx.DiGraph): The original migration network.
        """
        self.original_network = migration_network
        self.flow_network = None
        
    def build_flow_network(self, source_criteria_func, sink_criteria_func):
        """
        Constructs a flow network with node capacities via node splitting.
        
        Args:
            source_criteria_func (func): Function(node_data) -> bool. Returns true if node is a potential source.
            sink_criteria_func (func): Function(node_data) -> bool. Returns true if node is a potential sink.
            
        Returns:
            nx.DiGraph: The flow network.
        """
        print("Building flow network with node splitting...")
        G = nx.DiGraph()
        
        # Super Source and Super Sink
        S = "SUPER_SOURCE"
        T = "SUPER_SINK"
        G.add_node(S)
        G.add_node(T)
        
        # Process nodes
        source_count = 0
        sink_count = 0
        
        for node, data in self.original_network.nodes(data=True):
            # Node Splitting: u -> u_in, u_out
            u_in = f"{node}_in"
            u_out = f"{node}_out"
            
            capacity = data.get('total_cranes', 0)
            if capacity == 0:
                 capacity = 1 # Minimum flow to avoid zero capacity issues if needed
            
            # Add split nodes
            G.add_node(u_in, type='in', original=node, **data)
            G.add_node(u_out, type='out', original=node, **data)
            
            # Add internal edge with node capacity
            # Cost of passing through a node is 0? Or associated with stay? Assuming 0 for now.
            G.add_edge(u_in, u_out, capacity=capacity, weight=0)
            
            # Connect to Super Source/Sink if applicable
            if source_criteria_func(data):
                # Source connects to u_in with infinite capacity
                G.add_edge(S, u_in, capacity=float('inf'), weight=0)
                source_count += 1
                
            if sink_criteria_func(data):
                # u_out connects to Sink with infinite capacity
                G.add_edge(u_out, T, capacity=float('inf'), weight=0)
                sink_count += 1
                
        print(f"Selected {source_count} source nodes and {sink_count} sink nodes.")
                
        # Process edges from original network
        for u, v, data in self.original_network.edges(data=True):
            # Original edge (u, v) becomes (u_out, v_in)
            u_out = f"{u}_out"
            v_in = f"{v}_in"
            
            if u_out in G and v_in in G:
                # Edge capacity is infinite (constrained by nodes)
                # Edge cost is distance
                dist = data.get('distance', 1.0)
                G.add_edge(u_out, v_in, capacity=float('inf'), weight=dist)
                
        self.flow_network = G
        
        # Check if there's a path from source to sink
        if source_count > 0 and sink_count > 0:
            try:
                has_path = nx.has_path(G, S, T)
                print(f"Path exists from source to sink: {has_path}")
                if not has_path:
                    print("Warning: No path exists from source to sink. Flow will be 0.")
            except:
                pass
        
        print(f"Flow network built: {G.number_of_nodes()} nodes (split), {G.number_of_edges()} edges.")
        return G
        
    def calculate_max_flow(self):
        """
        Calculates max flow from Super Source to Super Sink.
        """
        if not self.flow_network:
            raise ValueError("Flow network not built. Call build_flow_network first.")
            
        print("Calculating Maximum Flow...")
        flow_value, flow_dict = nx.maximum_flow(self.flow_network, "SUPER_SOURCE", "SUPER_SINK", capacity='capacity')
        print(f"Maximum Flow Value: {flow_value}")
        
        return flow_value, flow_dict
        
    def calculate_min_cut(self):
        """
        Calculates minimum cut.
        """
        if not self.flow_network:
            raise ValueError("Flow network not built.")
            
        print("Calculating Minimum Cut...")
        cut_value, partition = nx.minimum_cut(self.flow_network, "SUPER_SOURCE", "SUPER_SINK", capacity='capacity')
        reachable, non_reachable = partition
        
        # Find edges that cross the cut
        cut_edges = []
        for u in reachable:
            for v in self.flow_network.adj[u]:
                if v in non_reachable:
                    cut_edges.append((u, v))
                    
        print(f"Min Cut Value: {cut_value}")
        print(f"Edges in Min Cut: {len(cut_edges)}")
        
        return cut_value, cut_edges
    
    def extract_bottleneck_locations(self, cut_edges):
        """
        Extract geographic locations of bottlenecks from min-cut edges.
        
        Returns:
            dict: {
                'node_bottlenecks': [{'node_id': ..., 'lat': ..., 'lon': ..., 'capacity': ...}],
                'edge_bottlenecks': [{'from': ..., 'to': ..., 'from_lat': ..., 'from_lon': ..., 
                                      'to_lat': ..., 'to_lon': ..., 'distance': ...}]
            }
        """
        node_bottlenecks = []
        edge_bottlenecks = []
        seen_nodes = set()  # Avoid duplicates
        
        for u_flow, v_flow in cut_edges:
            # Node bottleneck: internal edge (X_in -> X_out)
            if u_flow.endswith('_in') and v_flow.endswith('_out'):
                node_id = u_flow[:-3]  # Remove "_in"
                if node_id in self.original_network and node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    node_data = self.original_network.nodes[node_id]
                    node_bottlenecks.append({
                        'node_id': node_id,
                        'lat': node_data.get('lat'),
                        'lon': node_data.get('lon'),
                        'capacity': node_data.get('total_cranes', 0),
                        'type': 'node_bottleneck'
                    })
            
            # Edge bottleneck: travel edge (X_out -> Y_in)
            elif u_flow.endswith('_out') and v_flow.endswith('_in'):
                u_orig = u_flow[:-4]  # Remove "_out"
                v_orig = v_flow[:-3]  # Remove "_in"
                
                edge_key = (u_orig, v_orig)
                if u_orig in self.original_network and v_orig in self.original_network:
                    u_data = self.original_network.nodes[u_orig]
                    v_data = self.original_network.nodes[v_orig]
                    edge_data = self.original_network.edges.get((u_orig, v_orig), {})
                    
                    edge_bottlenecks.append({
                        'from_node': u_orig,
                        'to_node': v_orig,
                        'from_lat': u_data.get('lat'),
                        'from_lon': u_data.get('lon'),
                        'to_lat': v_data.get('lat'),
                        'to_lon': v_data.get('lon'),
                        'distance': edge_data.get('distance', 0),
                        'type': 'edge_bottleneck'
                    })
        
        return {
            'node_bottlenecks': node_bottlenecks,
            'edge_bottlenecks': edge_bottlenecks
        }

