import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        features['tx_interaction_ratio'] = (
            features['unique_interactions'] / features['transaction_count'].replace(0, 1)
        )

        features['balance_volatility'] = (
            features['max_balance'] / features['current_balance'].replace(0, 1)
        )

        features['contract_interaction_ratio'] = (
            features['contract_interaction_count'] / features['transaction_count'].replace(0, 1)
        )

        features['tx_value_cv'] = (
            features['transaction_value_std'] / features['avg_transaction_value'].replace(0, 1)
        )

        features['incoming_outgoing_ratio'] = (
            features['incoming_tx_count'] / features['outgoing_tx_count'].replace(0, 1)
        )

        features['account_age_days'] = (
            pd.Timestamp.now().timestamp() - features['creation_timestamp']
        ) / (24 * 3600)

        features['tx_frequency'] = (
            features['transaction_count'] / features['account_age_days'].replace(0, 1)
        )

        features['nft_activity_score'] = (
            features['nft_transfer_count'] * features['erc20_token_count']
        )

        features['diversity_score'] = (
            features['erc20_token_count'] +
            features['nft_transfer_count'] +
            features['contract_interaction_count']
        ) / features['transaction_count'].replace(0, 1)

        return features

    def extract_graph_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
        addresses = df['address'].tolist()

        in_degree = []
        out_degree = []
        pagerank_scores = []
        clustering_coeffs = []
        betweenness_centrality = []

        try:
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except:
            pagerank = {node: 0.0 for node in graph.nodes()}

        try:
            betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))
        except:
            betweenness = {node: 0.0 for node in graph.nodes()}

        undirected_graph = graph.to_undirected()
        try:
            clustering = nx.clustering(undirected_graph)
        except:
            clustering = {node: 0.0 for node in graph.nodes()}

        for addr in addresses:
            if addr in graph:
                in_degree.append(graph.in_degree(addr))
                out_degree.append(graph.out_degree(addr))
                pagerank_scores.append(pagerank.get(addr, 0.0))
                clustering_coeffs.append(clustering.get(addr, 0.0))
                betweenness_centrality.append(betweenness.get(addr, 0.0))
            else:
                in_degree.append(0)
                out_degree.append(0)
                pagerank_scores.append(0.0)
                clustering_coeffs.append(0.0)
                betweenness_centrality.append(0.0)

        df['in_degree'] = in_degree
        df['out_degree'] = out_degree
        df['total_degree'] = np.array(in_degree) + np.array(out_degree)
        df['pagerank'] = pagerank_scores
        df['clustering_coefficient'] = clustering_coeffs
        df['betweenness_centrality'] = betweenness_centrality

        df['degree_centrality'] = df['total_degree'] / max(1, len(addresses) - 1)

        return df

    def extract_cluster_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
        addresses = df['address'].tolist()

        same_cluster_connections = []
        cluster_density = []

        for addr in addresses:
            if addr not in graph:
                same_cluster_connections.append(0)
                cluster_density.append(0.0)
                continue

            neighbors = list(graph.neighbors(addr))

            if len(neighbors) == 0:
                same_cluster_connections.append(0)
                cluster_density.append(0.0)
                continue

            neighbor_neighbors = set()
            for neighbor in neighbors:
                if neighbor in graph:
                    neighbor_neighbors.update(graph.neighbors(neighbor))

            shared_connections = len(set(neighbors).intersection(neighbor_neighbors))
            same_cluster_connections.append(shared_connections)

            max_edges = len(neighbors) * (len(neighbors) - 1)
            if max_edges > 0:
                actual_edges = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if graph.has_edge(n1, n2) or graph.has_edge(n2, n1):
                            actual_edges += 1
                cluster_density.append(actual_edges / max_edges)
            else:
                cluster_density.append(0.0)

        df['same_cluster_connections'] = same_cluster_connections
        df['local_cluster_density'] = cluster_density

        return df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['tx_burst_score'] = (
            df['transaction_count'] / df['avg_time_between_tx'].replace(0, 1)
        )

        df['early_activity_score'] = np.where(
            df['account_age_days'] > 0,
            df['transaction_count'] / np.log1p(df['account_age_days']),
            0
        )

        return df

    def extract_all_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
        features = self.extract_basic_features(df)
        features = self.extract_graph_features(features, graph)
        features = self.extract_cluster_features(features, graph)
        features = self.extract_temporal_features(features)

        self.feature_names = [
            'transaction_count', 'unique_interactions', 'avg_transaction_value',
            'transaction_value_std', 'current_balance', 'max_balance',
            'gas_price_std', 'avg_time_between_tx', 'contract_interaction_count',
            'nft_transfer_count', 'erc20_token_count', 'incoming_tx_count',
            'outgoing_tx_count', 'tx_interaction_ratio', 'balance_volatility',
            'contract_interaction_ratio', 'tx_value_cv', 'incoming_outgoing_ratio',
            'account_age_days', 'tx_frequency', 'nft_activity_score',
            'diversity_score', 'in_degree', 'out_degree', 'total_degree',
            'pagerank', 'clustering_coefficient', 'betweenness_centrality',
            'degree_centrality', 'same_cluster_connections', 'local_cluster_density',
            'tx_burst_score', 'early_activity_score'
        ]

        return features

    def fit_transform(self, df: pd.DataFrame, graph: nx.DiGraph) -> np.ndarray:
        features = self.extract_all_features(df, graph)
        X = features[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def transform(self, df: pd.DataFrame, graph: nx.DiGraph) -> np.ndarray:
        features = self.extract_all_features(df, graph)
        X = features[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        return X_scaled


def main():
    df = pd.read_csv('data/wallet_dataset.csv')
    graph = nx.read_gpickle('data/transaction_graph.gpickle')

    extractor = FeatureExtractor()
    X = extractor.fit_transform(df, graph)

    print(f"Extracted features shape: {X.shape}")
    print(f"Feature names: {extractor.feature_names}")

    np.save('data/features.npy', X)


if __name__ == "__main__":
    main()
