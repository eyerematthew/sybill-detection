# Dataset Schema Documentation

## Overview

This document provides detailed specifications for the Sybil wallet detection dataset, including field definitions, data types, value ranges, and sourcing methodology.

## Dataset Files

### 1. `wallet_dataset.csv`

Primary dataset containing wallet-level features and labels.

**Size**: 7,000 rows (5,000 legitimate + 2,000 Sybil wallets)

**Format**: CSV with header row

#### Field Specifications

| Field Name | Type | Range | Description | Source |
|------------|------|-------|-------------|--------|
| `address` | string | 42 chars | Ethereum wallet address (0x + 40 hex chars) | Generated |
| `is_sybil` | int | {0, 1} | Binary label: 0=legitimate, 1=Sybil | Ground truth |
| `creation_timestamp` | float | Unix timestamp | Wallet first activity timestamp | Simulated |
| `transaction_count` | int | [5, 5000] | Total number of transactions | Simulated |
| `unique_interactions` | int | [3, 4750] | Number of unique counterparty addresses | Simulated |
| `avg_transaction_value` | float | [0.001, 100] ETH | Mean transaction value | Simulated |
| `transaction_value_std` | float | [0.0005, 200] ETH | Standard deviation of transaction values | Simulated |
| `current_balance` | float | [0, 500] ETH | Current wallet balance | Simulated |
| `max_balance` | float | [0, 2500] ETH | Historical maximum balance | Simulated |
| `gas_price_std` | float | [1, 50] Gwei | Standard deviation of gas prices used | Simulated |
| `avg_time_between_tx` | float | [0.5, 168] hours | Mean time interval between transactions | Simulated |
| `contract_interaction_count` | int | [0, 3000] | Number of smart contract calls | Simulated |
| `nft_transfer_count` | int | [0, 100] | Number of NFT transfers (ERC-721/1155) | Simulated |
| `erc20_token_count` | int | [0, 50] | Number of unique ERC-20 tokens held/traded | Simulated |
| `incoming_tx_count` | int | [0, 3500] | Number of incoming transactions | Simulated |
| `outgoing_tx_count` | int | [0, 3500] | Number of outgoing transactions | Simulated |
| `cluster_id` | int | [0, 100] | Cluster identifier (Sybil wallets only) | Generated |

#### Distribution Characteristics

**Legitimate Wallets:**
- Transaction count: Log-normal(μ=4, σ=1.5)
- Interaction diversity: 60-95% of transactions
- Account age: Uniform across 365 days
- Gas price variance: Moderate (5-30 Gwei std)
- Activity patterns: Exponential inter-transaction times

**Sybil Wallets:**
- Transaction count: Log-normal(μ=3, σ=0.5)
- Interaction diversity: 20-50% of transactions
- Account age: Clustered creation (within 72 hours)
- Gas price variance: Low (cluster-specific)
- Activity patterns: Burst-like (short inter-transaction times)

### 2. `transaction_graph.gpickle`

Network graph representing wallet interactions.

**Format**: NetworkX directed graph (pickled)

**Nodes**: Wallet addresses (7,000 total)

**Edges**: Transaction relationships

#### Edge Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | float | Transaction value in ETH |

#### Graph Statistics

- **Nodes**: 7,000
- **Edges**: ~50,000-80,000 (varies per generation)
- **Average Degree**: 14-23
- **Graph Density**: 0.001-0.002
- **Connected Components**: 1-5 major, multiple small

#### Structural Patterns

**Legitimate Subgraphs:**
- Random connectivity
- Lower clustering coefficient
- Diverse edge weights
- Broader degree distribution

**Sybil Subgraphs:**
- Dense intra-cluster connections
- High clustering coefficient
- Similar edge weights within cluster
- Bridge nodes to external wallets

## Data Generation Methodology

### Legitimate Wallet Generation

```python
Parameters:
  n = 5000
  tx_count ~ LogNormal(μ=4, σ=1.5)
  unique_ratio ~ Uniform(0.6, 0.95)
  value ~ LogNormal(μ=log(0.5), σ=1.5)
  gas_std ~ Uniform(5, 30)
  creation_time ~ Uniform(now-365d, now)
```

Key characteristics:
- Natural behavioral diversity
- Uncorrelated creation times
- Variable gas strategies
- Diverse DeFi interactions

### Sybil Wallet Generation

```python
Parameters:
  n = 2000 (across multiple clusters)
  cluster_sizes ~ Categorical([3, 5, 10, 20, 50])
  tx_count ~ LogNormal(μ=3, σ=0.5)
  unique_ratio ~ Uniform(0.2, 0.5)
  value ~ LogNormal(μ=log(0.1), σ=0.5)
  gas_std ~ Uniform(0.9, 1.1) × cluster_base
  creation_time ~ cluster_time + Uniform(0, 72h)
```

Key characteristics:
- Coordinated creation timestamps
- Similar transaction patterns within cluster
- Low interaction diversity
- Standardized gas prices

### Graph Construction

**Legitimate Edges:**
```python
For each wallet:
  n_edges = unique_interactions × 0.3
  targets = random_sample(all_wallets)
  weights ~ LogNormal(μ=log(avg_tx_value), σ=0.5)
```

**Sybil Edges:**
```python
Intra-cluster:
  connection_probability = 0.7
  weights ~ LogNormal(μ=log(cluster_avg), σ=0.2)

Inter-cluster:
  n_external = 5
  targets = random_sample(non_cluster_wallets)
  weights ~ LogNormal(μ=log(cluster_avg), σ=0.3)
```

## Quality Assurance

### Data Validation

All generated data undergoes validation:

1. **Range Checks**: All numeric fields within specified bounds
2. **Consistency Checks**:
   - `unique_interactions ≤ transaction_count`
   - `incoming + outgoing ≈ transaction_count`
   - `current_balance ≤ max_balance`
3. **Distribution Checks**: Statistical tests for expected distributions
4. **Graph Integrity**: Connected component analysis

### Label Accuracy

Ground truth labels are deterministically assigned during generation:
- Legitimate: Generated by `generate_legitimate_wallets()`
- Sybil: Generated by `generate_sybil_wallets()`

No labeling errors exist in synthetic data. Real-world deployment requires manual labeling or active learning.

## Feature Engineering

### Derived Features

Beyond raw fields, the system computes:

1. **Ratio Features**:
   - `tx_interaction_ratio = unique_interactions / transaction_count`
   - `balance_volatility = max_balance / current_balance`
   - `contract_interaction_ratio = contract_interactions / transaction_count`

2. **Graph Features**:
   - PageRank centrality
   - Clustering coefficient
   - Betweenness centrality
   - Local cluster density

3. **Temporal Features**:
   - Account age in days
   - Transaction frequency
   - Burst score

See `src/features/extractor.py` for complete feature definitions.

## Data Licensing

### License Type

**MIT License** - Permissive open-source license

### Usage Permissions

✓ Commercial use
✓ Modification
✓ Distribution
✓ Private use

### Conditions

- Include copyright notice
- Include license text

### Limitations

⚠ No warranty
⚠ No liability

### Full License Text

See `LICENSE` file in repository root.

## Citation

When using this dataset:

```bibtex
@dataset{sybil_wallet_dataset_2024,
  author = {Matthew Eyer},
  title = {Synthetic Sybil Wallet Detection Dataset},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/eyerematthew/sybil-wallet-detection}
}
```

## Data Privacy

This dataset contains **only synthetic data**. No real user information, addresses, or transactions are included.

For production deployment with real blockchain data:
- All blockchain data is already public
- Aggregate statistics and patterns only
- No PII collection
- Comply with applicable regulations (GDPR, CCPA)

## Reproducibility

The dataset is fully reproducible:

```python
from src.data.generator import WalletDataGenerator

generator = WalletDataGenerator(
    n_legitimate=5000,
    n_sybil=2000,
    seed=42
)
wallets_df, graph = generator.generate()
```

Using `seed=42` ensures identical dataset across runs.

## Data Freshness

**Generation Date**: On-demand
**Update Frequency**: Regenerated per training run
**Staleness**: Not applicable (synthetic data)

For production:
- Update monthly with new blockchain data
- Retrain quarterly with labeled examples
- Monitor for distribution shift

## Known Limitations

1. **Synthetic Nature**: May not capture all real-world patterns
2. **Simplified Graph**: Real blockchain graphs are more complex
3. **Single Chain**: Ethereum-only (multi-chain behavior differs)
4. **Static Snapshot**: No temporal evolution modeling
5. **Perfect Labels**: Real-world labeling has noise

## Contact

Questions about the dataset:
- GitHub Issues: https://github.com/eyerematthew/sybil-wallet-detection/issues
- Pond Profile: https://pond0x.com/@eyerematthew
