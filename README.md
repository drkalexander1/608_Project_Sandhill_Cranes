# Sandhill Crane Migration Network Analysis

## Overview

This project transforms sandhill crane observation data into a network dataset for analyzing migration patterns. The approach aggregates observations by FIPS county codes and creates edges based on temporal and geographic proximity, representing the flow of cranes between locations during migration.

## Dataset

The dataset contains sandhill crane observations from 2018-2023, with over 3.2 million individual observations across 6 years. Each observation includes:

- **Location data**: FIPS county codes, latitude/longitude coordinates
- **Temporal data**: Observation dates and times
- **Observer data**: Unique observer identifiers
- **Species data**: Scientific name (Antigone canadensis)
- **Count data**: Number of cranes observed per checklist

## Network Construction Approach

### Nodes
- **Geographic units**: Counties (FIPS codes)
- **Aggregation**: Multiple observations per county are aggregated
- **Attributes**: Total crane count, coordinates, number of observers, observation duration

### Edges
- **Connection criteria**: Counties within 500km and 14 days of each other
- **Weight calculation**: Proportional flow based on conservation of mass
- **Direction**: Directed edges representing migration flow direction

### Edge Weight Formula
```
Edge Weight = min(proportion_A, proportion_B) × temporal_decay × distance_decay
```

Where:
- `proportion_A/B`: Proportion of total cranes in each county
- `temporal_decay`: Exponential decay based on time gap
- `distance_decay`: Exponential decay based on geographic distance

## Key Features

### 1. Population Conservation
- **No bird creation**: Total crane population is conserved across the network
- **Proportional distribution**: Hidden/unobserved birds are distributed proportionally
- **Realistic flow**: Edge weights represent actual migration patterns

### 2. Geographic Constraints
- **Distance threshold**: Maximum 500km between connected counties
- **Migration distance**: Based on sandhill crane migration capabilities
- **Regional focus**: Prevents false connections across distant regions

### 3. Temporal Constraints
- **Time window**: Maximum 14 days between observations
- **Migration timing**: Captures seasonal migration patterns
- **Temporal decay**: Closer observations have stronger connections

### 4. Network Properties
- **Directed graph**: Edges represent migration flow direction
- **Weighted edges**: Edge weights represent flow strength
- **Node attributes**: Rich metadata for each county
- **Edge attributes**: Distance, time gap, flow direction

## Usage

### Basic Network Construction
```python
from network_transformation import CraneNetworkBuilder

# Initialize network builder
builder = CraneNetworkBuilder(
    max_distance_km=500,    # Maximum migration distance
    max_time_days=14       # Maximum time gap
)

# Build network from data
network = builder.build_network('data_file.parquet')

# Save network
builder.save_network('migration_network.graphml')
```

### Network Analysis
```python
import networkx as nx

# Load network
network = nx.read_graphml('migration_network.graphml')

# Calculate network metrics
degree_centrality = nx.degree_centrality(network)
betweenness_centrality = nx.betweenness_centrality(network)

# Find migration corridors
high_degree_nodes = [node for node, degree in degree_centrality.items() 
                    if degree > threshold]
```

## Output Files

The script generates three output files:

1. **`migration_network.graphml`**: NetworkX format for analysis
2. **`migration_network_edges.csv`**: Edge list with weights and attributes
3. **`migration_network_nodes.csv`**: Node attributes and metadata

## Network Analysis Applications

### 1. Migration Route Identification
- **Centrality measures**: Identify key migration hubs
- **Path analysis**: Find shortest migration routes
- **Flow analysis**: Identify major migration corridors

### 2. Seasonal Pattern Analysis
- **Temporal networks**: Compare spring vs fall migration
- **Seasonal centrality**: Identify important locations by season
- **Migration timing**: Track migration progression

### 3. Conservation Applications
- **Critical habitats**: Identify essential migration stopover sites
- **Bottleneck analysis**: Find locations with high flow but limited capacity
- **Route protection**: Prioritize conservation efforts

### 4. Population Dynamics
- **Flow conservation**: Verify population conservation across network
- **Migration efficiency**: Measure migration route effectiveness
- **Population distribution**: Understand spatial distribution patterns

## Technical Details

### Dependencies
- `pandas`: Data manipulation and analysis
- `networkx`: Network analysis
- `geopy`: Geographic distance calculations
- `numpy`: Numerical computations

### Performance Considerations
- **Memory usage**: Large datasets require significant RAM
- **Computation time**: Distance calculations scale as O(n²)
- **Network size**: Aggregation by county reduces complexity

### Limitations
- **Observer bias**: Network reflects observer behavior patterns
- **Temporal gaps**: Missing observations may break migration chains
- **Geographic aggregation**: County-level aggregation may miss local patterns

## Future Enhancements

1. **Multi-species networks**: Include other migratory species
2. **Dynamic networks**: Time-varying network structure
3. **Machine learning**: Predict migration patterns
4. **Visualization**: Interactive network maps
5. **Validation**: Compare with GPS tracking data

## Contact

For questions or collaboration opportunities, please contact the project team.

---

*This network transformation approach provides a powerful framework for analyzing large-scale migration patterns using citizen science data.*
