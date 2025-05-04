// Module implementing K-Means clustering for fraud detection. Handles the clustering algorithm and analysis of transaction data.
use std::error::Error;
use ndarray::Array2;
use linfa_clustering::KMeans;
use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use crate::csv_reader::Transaction;

// Represents the analysis results for a single cluster, containing statistical information about transactions
#[derive(Debug, Clone)]
pub struct ClusterAnalysis {
    //this section defines the size of the cluster
    pub size: usize,
    //this section defines the number of fraudulent transactions in the cluster
    pub fraud_count: usize,
    //this section defines the number of unique users in the cluster
    pub unique_users: usize,
    //this section defines the average features for the cluster
    pub avg_features: Vec<f64>,
    //this section defines the most common transaction type for the cluster
    pub most_common_tx_type: String,
    //this section defines the most common payment method for the cluster
    pub most_common_payment: String,
}

// Performs K-Means clustering on transaction data and analyzes the resulting clusters
// Inputs: Vector of transactions and number of clusters
// Outputs: Vector of ClusterAnalysis objects
// Key steps:
// 1. Convert transactions to feature vectors
// 2. Create dataset matrix
// 3. Perform K-Means clustering
// 4. Analyze each cluster for fraud patterns
pub fn analyze_clusters(transactions: &[Transaction], n_clusters: usize) -> Result<Vec<ClusterAnalysis>, Box<dyn Error>> {
    // Validate input data
    if transactions.is_empty() {
        return Ok(Vec::new());
    }

    // Convert transactions to feature vectors
    let features: Vec<Vec<f64>> = transactions.iter()
        .map(|tx| tx.to_feature_vector())
        .collect();

    // Create dataset matrix for clustering
    let n_features = features[0].len();
    let mut data = Array2::zeros((features.len(), n_features));
    for (i, feature_vec) in features.iter().enumerate() {
        for (j, &value) in feature_vec.iter().enumerate() {
            data[[i, j]] = value;
        }
    }

    // Perform K-Means clustering
    let dataset = Dataset::from(data);
    let model = KMeans::params(n_clusters)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)?;

    // Analyze each cluster
    let mut cluster_analyses = Vec::new();
    for cluster_idx in 0..n_clusters {
        // Get transactions belonging to current cluster
        let cluster_transactions: Vec<&Transaction> = transactions.iter()
            //this section predicts the cluster index for the transaction
            .zip(model.predict(&dataset).iter())
            .filter(|(_, &pred)| pred == cluster_idx)
            .map(|(tx, _)| tx)
            .collect();
        //this section checks if the cluster is empty
        if cluster_transactions.is_empty() {
            continue;
        }
        //this section calculates the size of the cluster  
        let size = cluster_transactions.len();
        //this section calculates the number of fraudulent transactions in the cluster
        let fraud_count = cluster_transactions.iter()
            .filter(|tx| tx.fraudulent == Some(1))
            .count();
        //this section calculates the number of unique users in the cluster
        let unique_users = cluster_transactions.iter()
            .map(|tx| tx.user_id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        //this section calculates the average features for the cluster
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
        //this section calculates the most common transaction type and payment method for the cluster
        let mut tx_type_counts = std::collections::HashMap::new();
        let mut payment_counts = std::collections::HashMap::new();
        for tx in &cluster_transactions {
            *tx_type_counts.entry(tx.transaction_type.clone()).or_insert(0) += 1;
            *payment_counts.entry(tx.payment_method.clone()).or_insert(0) += 1;
        }
        //this section calculates the most common transaction type and payment method for the cluster
        let most_common_tx_type = tx_type_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(tx_type, _)| tx_type)
            .unwrap_or_default();
        //this section calculates the most common payment method for the cluster
        let most_common_payment = payment_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(payment, _)| payment)
            .unwrap_or_default();
        //this section pushes the cluster analysis to the vector
        cluster_analyses.push(ClusterAnalysis {
            size,
            fraud_count,
            unique_users,
            avg_features,
            most_common_tx_type,
            most_common_payment,
        });
    }
    //this section returns the vector of cluster analyses   
    Ok(cluster_analyses)
}