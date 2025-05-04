use std::error::Error;
use ndarray::Array2;
use linfa_clustering::KMeans;
use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use crate::csv_reader::Transaction;

#[derive(Debug, Clone)]
pub struct ClusterAnalysis {
    pub size: usize,
    pub fraud_count: usize,
    pub unique_users: usize,
    pub avg_features: Vec<f64>,
    pub most_common_tx_type: String,
    pub most_common_payment: String,
}

pub fn analyze_clusters(transactions: &[Transaction], n_clusters: usize) -> Result<Vec<ClusterAnalysis>, Box<dyn Error>> {
    if transactions.is_empty() {
        return Ok(Vec::new());
    }

    let features: Vec<Vec<f64>> = transactions.iter()
        .map(|tx| tx.to_feature_vector())
        .collect();

    let n_features = features[0].len();
    let mut data = Array2::zeros((features.len(), n_features));
    for (i, feature_vec) in features.iter().enumerate() {
        for (j, &value) in feature_vec.iter().enumerate() {
            data[[i, j]] = value;
        }
    }

    let dataset = Dataset::from(data);

    let model = KMeans::params(n_clusters)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)?;

    let mut cluster_analyses = Vec::new();
    for cluster_idx in 0..n_clusters {
        let cluster_transactions: Vec<&Transaction> = transactions.iter()
            .zip(model.predict(&dataset).iter())
            .filter(|(_, &pred)| pred == cluster_idx)
            .map(|(tx, _)| tx)
            .collect();

        if cluster_transactions.is_empty() {
            continue;
        }

        let size = cluster_transactions.len();
        let fraud_count = cluster_transactions.iter()
            .filter(|tx| tx.fraudulent == Some(1))
            .count();
        let unique_users = cluster_transactions.iter()
            .map(|tx| tx.user_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let mut avg_features = vec![0.0; n_features];
        for tx in &cluster_transactions {
            let features = tx.to_feature_vector();
            for (i, &value) in features.iter().enumerate() {
                avg_features[i] += value;
            }
        }
        for value in &mut avg_features {
            *value /= size as f64;
        }

        let mut tx_type_counts = std::collections::HashMap::new();
        let mut payment_counts = std::collections::HashMap::new();
        for tx in &cluster_transactions {
            *tx_type_counts.entry(tx.transaction_type.clone()).or_insert(0) += 1;
            *payment_counts.entry(tx.payment_method.clone()).or_insert(0) += 1;
        }

        let most_common_tx_type = tx_type_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(tx_type, _)| tx_type)
            .unwrap_or_default();

        let most_common_payment = payment_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(payment, _)| payment)
            .unwrap_or_default();

        cluster_analyses.push(ClusterAnalysis {
            size,
            fraud_count,
            unique_users,
            avg_features,
            most_common_tx_type,
            most_common_payment,
        });
    }

    Ok(cluster_analyses)
}