# Style Cluster Fit Summary

- Chosen k: `4`
- Selection reason: `highest silhouette score on train-writer PCA embeddings`
- Train writer count: `75`
- PCA components: `8`

## Candidate k Results

- k=`4` silhouette=`0.1958` inertia=`1194.8674`
- k=`5` silhouette=`0.1704` inertia=`1069.7893`
- k=`6` silhouette=`0.1613` inertia=`995.3954`
- k=`7` silhouette=`0.1856` inertia=`912.9436`
- k=`8` silhouette=`0.1701` inertia=`865.2611`
- k=`9` silhouette=`0.1538` inertia=`798.3319`
- k=`10` silhouette=`0.1550` inertia=`780.3175`
- k=`11` silhouette=`0.1782` inertia=`695.7596`
- k=`12` silhouette=`0.1745` inertia=`659.7355`

## Split Cluster Coverage

- `train`: sample_count=`10525` writer_count=`75`
  sample_count_by_cluster=`{'0': 3097, '1': 444, '2': 1383, '3': 5601}` writer_count_by_cluster=`{'0': 24, '1': 6, '2': 11, '3': 34}`
- `val`: sample_count=`1234` writer_count=`9`
  sample_count_by_cluster=`{'0': 713, '3': 521}` writer_count_by_cluster=`{'0': 5, '3': 4}`
- `test`: sample_count=`1271` writer_count=`10`
  sample_count_by_cluster=`{'0': 641, '1': 58, '2': 105, '3': 467}` writer_count_by_cluster=`{'0': 5, '1': 1, '2': 2, '3': 2}`

## Artifacts

- Feature table: `configs/style_clusters/writer_style_features.csv`
- Train feature table: `configs/style_clusters/writer_style_features_train.csv`
- Writer to cluster map: `configs/style_clusters/writer_to_cluster_map.json`
- Cluster centroids: `configs/style_clusters/cluster_centroids.json`
- Default panel clusters: `configs/style_clusters/default_panel_clusters.json`
- Top writers by cluster: `configs/style_clusters/cluster_top_writers.csv`