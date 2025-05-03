use std::collections::HashMap;
use ndarray::Array;
use linfa_clustering::KMeans;
use linfa::traits::Fit;
use linfa::prelude::Predict;
use linfa::Dataset;
use crate::csv_reader::Transaction;

// Normalize a vector of features
fn normalize_features(features: &mut [f64]) {
    // Calculate mean and standard deviation
    let n = features.len() as f64;
    let mean = features.iter().sum::<f64>() / n;
    let variance = features.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();
    
    // Normalize each feature
    if std_dev > 0.0 {
        for x in features {
            *x = (*x - mean) / std_dev;
        }
    }
}

pub struct ClusterAnalysis {
    pub size: usize,
    pub fraud_count: usize,
    pub unique_users: usize,
    pub avg_features: Vec<f64>,
    pub most_common_tx_type: String,
    pub most_common_payment: String,
}

pub fn analyze_clusters(
    transactions: &[Transaction],
    n_clusters: usize,
) -> Result<Vec<ClusterAnalysis>, Box<dyn std::error::Error>> {
    // Prepare features for k-means
    let mut features: Vec<Vec<f64>> = transactions.iter()
        .map(|tx| tx.to_feature_vector())
        .collect();
    
    let n_features = features[0].len();
    let n_samples = features.len();
    
    // Normalize each feature column
    for i in 0..n_features {
        let mut column: Vec<f64> = features.iter().map(|f| f[i]).collect();
        normalize_features(&mut column);
        for (j, &value) in column.iter().enumerate() {
            features[j][i] = value;
        }
    }
    
    // Convert to ndarray
    let mut data = Array::zeros((n_samples, n_features));
    for (i, feature) in features.iter().enumerate() {
        for (j, &value) in feature.iter().enumerate() {
            data[[i, j]] = value;
        }
    }
    
    // Create dataset for k-means
    let dataset = Dataset::from(data);
    
    // Run k-means clustering
    let model = KMeans::params(n_clusters)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)?;
    
    let labels = model.predict(&dataset);
    
    // Analyze clusters
    let mut cluster_analysis = HashMap::new();
    let mut cluster_features = HashMap::new();
    
    // Initialize feature accumulators for each cluster
    for i in 0..n_clusters {
        cluster_features.insert(i, vec![0.0; n_features]);
    }
    
    // Accumulate features and count transactions
    for (i, &label) in labels.iter().enumerate() {
        let entry = cluster_analysis.entry(label).or_insert((0, 0, Vec::new()));
        entry.0 += 1;
        if transactions[i].fraudulent == Some(1) {
            entry.1 += 1;
        }
        entry.2.push(transactions[i].user_id);
        
        // Add features to cluster accumulator
        let features = &mut cluster_features.get_mut(&label).unwrap();
        let tx_features = transactions[i].to_feature_vector();
        for (j, &value) in tx_features.iter().enumerate() {
            features[j] += value;
        }
    }
    
    // Convert to Vec<ClusterAnalysis>
    let mut results = Vec::new();
    for (label, (size, fraud_count, users)) in cluster_analysis {
        let features = &cluster_features[&label];
        let avg_features: Vec<f64> = features.iter()
            .map(|&x| x / size as f64)
            .collect();
        
        let tx_type = avg_features[7];
        let payment = avg_features[8];
        
        results.push(ClusterAnalysis {
            size,
            fraud_count,
            unique_users: users.into_iter().collect::<std::collections::HashSet<_>>().len(),
            avg_features,
            most_common_tx_type: match tx_type {
                x if x < 0.5 => "ATM Withdrawal".to_string(),
                x if x < 1.5 => "Bill Payment".to_string(),
                x if x < 2.5 => "Online Purchase".to_string(),
                x if x < 3.5 => "Transfer".to_string(),
                _ => "Other".to_string(),
            },
            most_common_payment: match payment {
                x if x < 0.5 => "Credit Card".to_string(),
                x if x < 1.5 => "Debit Card".to_string(),
                x if x < 2.5 => "PayPal".to_string(),
                x if x < 3.5 => "Bank Transfer".to_string(),
                _ => "Other".to_string(),
            },
        });
    }
    
    Ok(results)
} 