import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import networkx as nx
from typing import Dict, List, Tuple
import random


class WalletDataGenerator:

    def __init__(self, n_legitimate: int = 5000, n_sybil: int = 2000, seed: int = 42):
        self.n_legitimate = n_legitimate
        self.n_sybil = n_sybil
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.start_date = datetime.now() - timedelta(days=365)

    def generate_wallet_address(self, index: int) -> str:
        return f"0x{random.randbytes(20).hex()}"

    def generate_legitimate_wallets(self) -> pd.DataFrame:
        wallets = []

        for i in range(self.n_legitimate):
            creation_time = self.start_date + timedelta(
                days=np.random.uniform(0, 365)
            )

            tx_count = int(np.random.lognormal(4, 1.5))
            tx_count = max(10, min(tx_count, 5000))

            unique_interactions = int(tx_count * np.random.uniform(0.6, 0.95))

            avg_tx_value = np.random.lognormal(np.log(0.5), 1.5)
            tx_value_std = avg_tx_value * np.random.uniform(0.3, 2.0)

            balance = np.random.lognormal(np.log(2), 2)

            gas_price_std = np.random.uniform(5, 30)

            time_between_tx_hours = np.random.exponential(24 * 7)

            contract_interactions = int(tx_count * np.random.uniform(0.1, 0.6))

            nft_transfers = int(np.random.poisson(5))

            erc20_tokens = int(np.random.poisson(8))

            wallet = {
                'address': self.generate_wallet_address(i),
                'is_sybil': 0,
                'creation_timestamp': creation_time.timestamp(),
                'transaction_count': tx_count,
                'unique_interactions': unique_interactions,
                'avg_transaction_value': avg_tx_value,
                'transaction_value_std': tx_value_std,
                'current_balance': balance,
                'max_balance': balance * np.random.uniform(1.2, 5.0),
                'gas_price_std': gas_price_std,
                'avg_time_between_tx': time_between_tx_hours,
                'contract_interaction_count': contract_interactions,
                'nft_transfer_count': nft_transfers,
                'erc20_token_count': erc20_tokens,
                'incoming_tx_count': int(tx_count * np.random.uniform(0.3, 0.7)),
                'outgoing_tx_count': tx_count - int(tx_count * np.random.uniform(0.3, 0.7)),
            }

            wallets.append(wallet)

        return pd.DataFrame(wallets)

    def generate_sybil_clusters(self) -> List[List[int]]:
        clusters = []
        remaining = self.n_sybil

        while remaining > 0:
            cluster_size = min(
                remaining,
                int(np.random.choice([3, 5, 10, 20, 50], p=[0.1, 0.3, 0.3, 0.2, 0.1]))
            )
            clusters.append(list(range(len(clusters) * 50, len(clusters) * 50 + cluster_size)))
            remaining -= cluster_size

        return clusters

    def generate_sybil_wallets(self) -> pd.DataFrame:
        wallets = []
        clusters = self.generate_sybil_clusters()

        wallet_idx = 0
        for cluster in clusters:
            cluster_creation_time = self.start_date + timedelta(
                days=np.random.uniform(0, 300)
            )

            cluster_tx_pattern = np.random.lognormal(3, 0.5)
            cluster_value_pattern = np.random.lognormal(np.log(0.1), 0.5)
            cluster_gas_price = np.random.uniform(10, 25)

            for _ in cluster:
                creation_time = cluster_creation_time + timedelta(
                    hours=np.random.uniform(0, 72)
                )

                tx_count = int(cluster_tx_pattern * np.random.uniform(0.8, 1.2))
                tx_count = max(5, min(tx_count, 200))

                unique_interactions = int(tx_count * np.random.uniform(0.2, 0.5))

                avg_tx_value = cluster_value_pattern * np.random.uniform(0.9, 1.1)
                tx_value_std = avg_tx_value * np.random.uniform(0.1, 0.3)

                balance = np.random.lognormal(np.log(0.1), 0.8)

                gas_price_std = cluster_gas_price * np.random.uniform(0.9, 1.1)

                time_between_tx_hours = np.random.exponential(2)

                contract_interactions = int(tx_count * np.random.uniform(0.05, 0.2))

                wallet = {
                    'address': self.generate_wallet_address(wallet_idx),
                    'is_sybil': 1,
                    'creation_timestamp': creation_time.timestamp(),
                    'transaction_count': tx_count,
                    'unique_interactions': unique_interactions,
                    'avg_transaction_value': avg_tx_value,
                    'transaction_value_std': tx_value_std,
                    'current_balance': balance,
                    'max_balance': balance * np.random.uniform(1.05, 1.5),
                    'gas_price_std': gas_price_std,
                    'avg_time_between_tx': time_between_tx_hours,
                    'contract_interaction_count': contract_interactions,
                    'nft_transfer_count': int(np.random.poisson(0.5)),
                    'erc20_token_count': int(np.random.poisson(2)),
                    'incoming_tx_count': int(tx_count * np.random.uniform(0.4, 0.6)),
                    'outgoing_tx_count': tx_count - int(tx_count * np.random.uniform(0.4, 0.6)),
                    'cluster_id': len(wallets) // max(1, len(cluster))
                }

                wallets.append(wallet)
                wallet_idx += 1

        return pd.DataFrame(wallets)

    def generate_transaction_graph(self, wallets_df: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()

        addresses = wallets_df['address'].tolist()
        G.add_nodes_from(addresses)

        for idx, row in wallets_df.iterrows():
            if row['is_sybil'] == 0:
                n_edges = int(row['unique_interactions'] * 0.3)
                targets = np.random.choice(addresses, size=min(n_edges, len(addresses)), replace=False)

                for target in targets:
                    weight = np.random.lognormal(np.log(row['avg_transaction_value']), 0.5)
                    G.add_edge(row['address'], target, weight=weight)
            else:
                cluster_id = row.get('cluster_id', 0)
                cluster_wallets = wallets_df[
                    (wallets_df['is_sybil'] == 1) &
                    (wallets_df.get('cluster_id', -1) == cluster_id)
                ]['address'].tolist()

                for target in cluster_wallets:
                    if target != row['address'] and np.random.random() < 0.7:
                        weight = np.random.lognormal(np.log(row['avg_transaction_value']), 0.2)
                        G.add_edge(row['address'], target, weight=weight)

                external_targets = np.random.choice(
                    [a for a in addresses if a not in cluster_wallets],
                    size=min(5, len(addresses) - len(cluster_wallets)),
                    replace=False
                )
                for target in external_targets:
                    weight = np.random.lognormal(np.log(row['avg_transaction_value']), 0.3)
                    G.add_edge(row['address'], target, weight=weight)

        return G

    def generate(self) -> Tuple[pd.DataFrame, nx.DiGraph]:
        legitimate_df = self.generate_legitimate_wallets()
        sybil_df = self.generate_sybil_wallets()

        wallets_df = pd.concat([legitimate_df, sybil_df], ignore_index=True)
        wallets_df = wallets_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        graph = self.generate_transaction_graph(wallets_df)

        return wallets_df, graph


def main():
    generator = WalletDataGenerator(n_legitimate=5000, n_sybil=2000)
    wallets_df, graph = generator.generate()

    wallets_df.to_csv('data/wallet_dataset.csv', index=False)
    nx.write_gpickle(graph, 'data/transaction_graph.gpickle')

    print(f"Generated {len(wallets_df)} wallets")
    print(f"Legitimate: {(wallets_df['is_sybil'] == 0).sum()}")
    print(f"Sybil: {(wallets_df['is_sybil'] == 1).sum()}")
    print(f"Graph edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
