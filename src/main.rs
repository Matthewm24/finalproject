// Main module for fraud detection using K-Means clustering. Orchestrates data loading, analysis, and result presentation.
use std::error::Error;
use csv_reader::read_transactions;
use kmeans::{analyze_clusters, ClusterAnalysis};

//imports other modules in finalproject file
mod csv_reader;
mod kmeans;
//test module
#[cfg(test)]
mod tests;

const CSV_FILE_PATH: &str = "fraud_detection.csv";
const N_CLUSTERS: usize = 10;
const HIGH_FRAUD_RATE_THRESHOLD: f64 = 0.5;
const MEDIUM_FRAUD_RATE_THRESHOLD: f64 = 0.2;

// Holds aggregated metrics for all clusters including total transactions and fraud statistics
struct ClusterMetrics {
    total_transactions: usize,
    total_fraud: usize,
    fraud_rate: f64,
}

// Analyzes and displays detailed information about a single cluster
// Inputs: cluster analysis data, rank, and fraud rate
// Outputs: Prints formatted analysis to console
// Key steps: 
// 1. Display basic cluster statistics
// 2. Calculate and show average feature values
// 3. Identify most common transaction characteristics
// 4. Determine risk level based on fraud rate
fn print_cluster_analysis(cluster: &ClusterAnalysis, rank: usize, fraud_rate: f64) {
    // Display cluster rank and basic statistics
    println!("\nCluster Rank {} (Fraud Rate: {:.1}%)", rank, fraud_rate * 100.0);
    println!("Size: {} transactions", cluster.size);
    println!("Fraudulent: {} ({:.1}%)", cluster.fraud_count, fraud_rate * 100.0);
    println!("Unique Users: {}", cluster.unique_users);

    // Calculate and display average feature values
    println!("\nFeature Analysis:");
    if cluster.avg_features.len() == 5 {
        println!("Avg Amount: ${:.2}", cluster.avg_features[0]);
        println!("Avg Time: {:.1}", cluster.avg_features[1]);
        println!("Avg Prev. Fraud: {:.2}", cluster.avg_features[2]);
        println!("Avg Acct Age: {:.0} days", cluster.avg_features[3]);
        println!("Avg Recent Transactions: {:.1}", cluster.avg_features[4]);
    }

    // Display transaction characteristics
    println!("\nCommon Characteristics:");
    println!("Transaction Type: {}", cluster.most_common_tx_type);
    println!("Payment Method: {}", cluster.most_common_payment);

    // Determine and display risk level
    let risk_level = if fraud_rate >= HIGH_FRAUD_RATE_THRESHOLD {"High Risk"}
                     else if fraud_rate >= MEDIUM_FRAUD_RATE_THRESHOLD {"Medium Risk"}
                     else { "Low Risk" };
    println!("\n  Risk Level: {}", risk_level);
}

// Calculates overall metrics across all clusters
// Inputs: Vector of cluster analyses with their fraud rates
// Outputs: ClusterMetrics containing aggregated statistics
// Key steps:
// 1. Sum total transactions and fraud counts
// 2. Calculate overall fraud rate
fn calculate_metrics(clusters: &[(ClusterAnalysis, f64)]) -> ClusterMetrics {
    // Aggregate statistics across all clusters
    let total_transactions = clusters.iter().map(|(c, _)| c.size).sum();
    let total_fraud = clusters.iter().map(|(c, _)| c.fraud_count).sum();
    let fraud_rate = if total_transactions > 0 {
        total_fraud as f64 / total_transactions as f64
    } else {
        0.0
    };
    
    ClusterMetrics {
        total_transactions,
        total_fraud,
        fraud_rate,
    }
}

// Main entry point for the fraud detection system
// Inputs: None
// Outputs: Result indicating success or error
// Key steps:
// 1. Load transaction data
// 2. Perform clustering
// 3. Analyze and sort clusters
// 4. Display results
fn main() -> Result<(), Box<dyn Error>> {
    // Load and validate transaction data
    let transactions = read_transactions(CSV_FILE_PATH)?;
    if transactions.is_empty() {
        return Ok(());
    }

    // Perform K-Means clustering
    println!("K-Means clustering (k={})", N_CLUSTERS);
    let cluster_results = analyze_clusters(&transactions, N_CLUSTERS)?;
    if cluster_results.is_empty() {
        println!("Clustering resulted in no clusters to analyze.");
        return Ok(());
    }

    // Process and analyze clusters
    println!("Fraud Detection Analysis Results:");
    let mut analyzed_clusters: Vec<(ClusterAnalysis, f64)> = cluster_results.into_iter()
        .map(|cluster| {
            let fraud_rate = cluster.fraud_count as f64 / cluster.size.max(1) as f64;
            (cluster, fraud_rate)
        })
        .collect();

    // Sort clusters by fraud rate
    analyzed_clusters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Display results
    println!("\nHigh-Risk Clusters (Sorted by Fraud Rate):");
    for (i, (cluster, fraud_rate)) in analyzed_clusters.iter().enumerate() {
        print_cluster_analysis(cluster, i + 1, *fraud_rate);
    }

    // Display overall metrics
    let metrics = calculate_metrics(&analyzed_clusters);
    println!("Overall Metrics:");
    println!("Total Transactions: {}", metrics.total_transactions);
    println!("Total Fraudulent: {}", metrics.total_fraud);
    println!("Overall Fraud Rate: {:.2}%", metrics.fraud_rate * 100.0);

    Ok(())
}