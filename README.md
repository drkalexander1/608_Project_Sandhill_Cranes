# Sandhill Crane Migration Network Analysis

## Overview

This project transforms sandhill crane observation data (2018–2023) into directed, weighted networks to analyze migration patterns. Using over 3.2 million citizen science observations, it builds networks of migration flow between counties, then applies max-flow/min-cut analysis, community detection, and bottleneck prediction to identify key migration corridors and stopover sites.

## Data Setup

Input data files are not included in this repository due to size. Place the following parquet files in the project root before running:

```
Antigone canadensis_all_checklist_2018.parquet
Antigone canadensis_all_checklist_2019.parquet
Antigone canadensis_all_checklist_2020.parquet
Antigone canadensis_all_checklist_2021.parquet
Antigone canadensis_all_checklist_2022.parquet
Antigone canadensis_all_checklist_2023.parquet
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full pipeline (per-year, seasonal breakdown, bottleneck & community analysis)

```bash
python run_pipeline.py
```

This runs the complete `src/` pipeline: loads data, splits by season (PreBreeding/PostBreeding), builds county-level and cluster-level networks, performs max-flow min-cut analysis, detects communities, predicts bottlenecks, and generates interactive maps and summary plots. Results are written to `results/`.

### Multi-year network generation (efficient, precomputed distances)

```bash
python run_efficient_analysis.py
```

This first builds a reusable county distance table (`county_distance_table.pkl`), then generates migration networks for all 6 years with PNG and HTML map output.

## Repository Structure

```
├── run_pipeline.py               # Full seasonal pipeline entry point
├── run_efficient_analysis.py     # Multi-year efficient pipeline entry point
├── multi_year_migration_analysis.py  # Multi-year analyzer (used by above)
│
├── src/                          # Core library modules
│   ├── data_processing.py        # Data loading, filtering, county aggregation
│   ├── season_detection.py       # PreBreeding / PostBreeding season splitting
│   ├── node_generation.py        # County (FIPS) and cluster node strategies
│   ├── network_building.py       # Directed graph construction
│   ├── network_transformation.py # Network builder using precomputed distances
│   ├── distance_table.py         # Precomputes and caches county pair distances
│   ├── analysis.py               # Max-flow min-cut, bottleneck extraction
│   ├── visualization.py          # Folium maps, flow intensity, density plots
│   ├── bottleneck_predictor.py   # Random forest bottleneck prediction
│   └── community_detection.py    # Louvain / greedy community detection
│
├── scripts/                      # Standalone utility scripts
│   ├── generate_summary_plots.py      # Regenerate plots from existing GraphML files
│   ├── plot_birders.py                # Observer participation over time
│   ├── clean_geographic_map.py        # geopandas geographic map (2018)
│   ├── simple_folium_png_generator.py # Folium + Selenium PNG maps for all years
│   └── us_map_visualization.py        # Interactive Folium map (2018)
│
├── results/
│   ├── main/      # Key outputs: interactive maps, summary plots, birder stats
│   └── detailed/  # Full outputs: GraphML networks, per-component CSV tables
│
├── requirements.txt
└── .gitignore
```

All scripts in `scripts/` should be run from the project root:

```bash
python scripts/generate_summary_plots.py
python scripts/plot_birders.py
```

## Network Construction

### Nodes

- **County-based**: Observations aggregated by FIPS county code
- **Cluster-based**: HDBSCAN/DBSCAN geographic clusters as an alternative

### Edges

Directed edges are created between counties within **500 km** and **14 days** of each other.

**Edge weight formula:**

```
weight = min(proportion_A, proportion_B) × temporal_decay × distance_decay
```

Where:
- `proportion_A/B` — each county's share of the combined crane count
- `temporal_decay = exp(-time_gap / 14)`
- `distance_decay = exp(-distance / 500)`

This conserves total population and ensures closer, more contemporaneous observations produce stronger connections.

## Analysis Methods

| Method | Description |
|---|---|
| Max-flow / min-cut | Identifies migration bottlenecks: edges whose removal most reduces flow |
| Community detection | Louvain algorithm to find regional migration communities |
| Bottleneck prediction | Random forest classifier trained on network features to predict future bottlenecks |
| Seasonal comparison | Separate networks for PreBreeding (spring) and PostBreeding (fall) migration |

## Output Files

| Location | Contents |
|---|---|
| `results/main/` | Interactive HTML maps, flow intensity PNGs, network density PNGs, birder statistics |
| `results/detailed/` | GraphML network files, per-component bottleneck/community CSVs, summary statistics |

## Limitations

- **Observer bias**: Network density reflects eBird participation patterns as well as actual crane distribution
- **Temporal gaps**: Missing observations can break migration chains
- **Geographic aggregation**: County-level aggregation may mask fine-scale movement
