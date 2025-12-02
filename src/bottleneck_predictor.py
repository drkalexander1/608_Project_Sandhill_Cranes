import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class BottleneckPredictor:
    """
    Tracks bottlenecks across years and predicts future bottlenecks.
    """
    
    def __init__(self):
        self.historical_bottlenecks = []
        
    def add_year_data(self, year, season, bottleneck_locations, network):
        """
        Add bottleneck data for a year/season.
        
        Args:
            year: Year of the data
            season: Season name (PreBreeding/PostBreeding)
            bottleneck_locations: Dict with 'node_bottlenecks' and 'edge_bottlenecks'
            network: NetworkX graph
        """
        bottleneck_node_ids = {bn['node_id'] for bn in bottleneck_locations['node_bottlenecks']}
        
        # Calculate network metrics once
        betweenness = nx.betweenness_centrality(network)
        
        # Add bottleneck nodes
        for node_bn in bottleneck_locations['node_bottlenecks']:
            node_id = node_bn['node_id']
            if node_id not in network:
                continue
                
            node_data = network.nodes[node_id]
            in_degree = network.in_degree(node_id)
            out_degree = network.out_degree(node_id)
            
            self.historical_bottlenecks.append({
                'year': year,
                'season': season,
                'node_id': node_id,
                'lat': node_bn['lat'],
                'lon': node_bn['lon'],
                'capacity': node_bn['capacity'],
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness': betweenness.get(node_id, 0),
                'is_bottleneck': 1
            })
        
        # Sample non-bottleneck nodes for training (to balance dataset)
        non_bottleneck_nodes = [n for n in network.nodes() if n not in bottleneck_node_ids]
        # Sample up to 3x the number of bottlenecks
        sample_size = min(len(non_bottleneck_nodes), len(bottleneck_locations['node_bottlenecks']) * 3)
        sampled_nodes = np.random.choice(non_bottleneck_nodes, size=sample_size, replace=False) if sample_size > 0 else []
        
        for node_id in sampled_nodes:
            node_data = network.nodes[node_id]
            in_degree = network.in_degree(node_id)
            out_degree = network.out_degree(node_id)
            
            self.historical_bottlenecks.append({
                'year': year,
                'season': season,
                'node_id': node_id,
                'lat': node_data.get('lat', 0),
                'lon': node_data.get('lon', 0),
                'capacity': node_data.get('total_cranes', 0),
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness': betweenness.get(node_id, 0),
                'is_bottleneck': 0
            })
    
    def predict_future_bottlenecks(self, future_network, year, season):
        """
        Predict bottlenecks for a future network based on historical patterns.
        
        Args:
            future_network: NetworkX graph for future year
            year: Year to predict for
            season: Season name
            
        Returns:
            List of predicted bottlenecks with probabilities
        """
        if len(self.historical_bottlenecks) < 10:
            print(f"Not enough historical data for prediction (only {len(self.historical_bottlenecks)} samples)")
            return []
        
        # Prepare training data
        df = pd.DataFrame(self.historical_bottlenecks)
        features = ['lat', 'lon', 'capacity', 'in_degree', 'out_degree', 'betweenness']
        
        # Filter to same season if possible
        season_df = df[df['season'] == season] if season in df['season'].values else df
        
        if len(season_df) < 5:
            season_df = df  # Use all data if not enough for season
        
        X_train = season_df[features]
        y_train = season_df['is_bottleneck']
        
        if y_train.sum() == 0:
            print("No bottleneck examples in training data")
            return []
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Extract features from future network
        future_features = []
        future_nodes = []
        
        betweenness = nx.betweenness_centrality(future_network)
        
        for node_id in future_network.nodes():
            node_data = future_network.nodes[node_id]
            in_degree = future_network.in_degree(node_id)
            out_degree = future_network.out_degree(node_id)
            
            future_features.append([
                node_data.get('lat', 0),
                node_data.get('lon', 0),
                node_data.get('total_cranes', 0),
                in_degree,
                out_degree,
                betweenness.get(node_id, 0)
            ])
            future_nodes.append(node_id)
        
        # Predict
        X_future = pd.DataFrame(future_features, columns=features)
        predictions = model.predict(X_future)
        probabilities = model.predict_proba(X_future)[:, 1]
        
        # Return predicted bottlenecks (top 20% by probability)
        predicted = []
        for i, (node_id, pred, prob) in enumerate(zip(future_nodes, predictions, probabilities)):
            if pred == 1 or prob > 0.5:  # Include high-probability predictions
                node_data = future_network.nodes[node_id]
                predicted.append({
                    'node_id': node_id,
                    'lat': node_data.get('lat'),
                    'lon': node_data.get('lon'),
                    'capacity': node_data.get('total_cranes', 0),
                    'probability': prob,
                    'year': year,
                    'season': season
                })
        
        return sorted(predicted, key=lambda x: x['probability'], reverse=True)

