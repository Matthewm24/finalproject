use std::error::Error;
use csv_reader::read_transactions;
use kmeans::{analyze_clusters, ClusterAnalysis};

mod csv_reader;
mod kmeans;
#[cfg(test)]
mod tests;

const CSV_FILE_PATH: &str = "fraud_detection.csv";
const N_CLUSTERS: usize = 10;
const HIGH_FRAUD_RATE_THRESHOLD: f64 = 0.5;
const MEDIUM_FRAUD_RATE_THRESHOLD: f64 = 0.2;

struct ClusterMetrics {
    total_transactions: usize,
    total_fraud: usize,
    fraud_rate: f64,
}

fn print_cluster_analysis(cluster: &ClusterAnalysis, rank: usize, fraud_rate: f64) {
    println!("\nCluster Rank {} (Fraud Rate: {:.1}%)", rank, fraud_rate * 100.0);
    println!("Size: {} transactions", cluster.size);
    println!("Fraudulent: {} ({:.1}%)", cluster.fraud_count, fraud_rate * 100.0);
    println!("Unique Users: {}", cluster.unique_users);

    println!("\nFeature Analysis:");
    if cluster.avg_features.len() == 5 {
        println!("Avg Amount: ${:.2}", cluster.avg_features[0]);
        println!("Avg Time: {:.1}", cluster.avg_features[1]);
        println!("Avg Prev. Fraud: {:.2}", cluster.avg_features[2]);
        println!("Avg Acct Age: {:.0} days", cluster.avg_features[3]);
    }

    println!("\nCommon Characteristics:");
    println!("Transaction Type: {}", cluster.most_common_tx_type);
    println!("Payment Method: {}", cluster.most_common_payment);

    let risk_level = if fraud_rate >= HIGH_FRAUD_RATE_THRESHOLD {"High Risk"}
                     else if fraud_rate >= MEDIUM_FRAUD_RATE_THRESHOLD {"Medium Risk"}
                     else { "Low Risk" };
    println!("\n  Risk Level: {}", risk_level);
}

fn calculate_metrics(clusters: &[(ClusterAnalysis, f64)]) -> ClusterMetrics {
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

fn main() -> Result<(), Box<dyn Error>> {
    let transactions = read_transactions(CSV_FILE_PATH)?;

    if transactions.is_empty() {
        return Ok(());
    }

    println!("K-Means clustering (k={})", N_CLUSTERS);
    let cluster_results = analyze_clusters(&transactions, N_CLUSTERS)?;

    if cluster_results.is_empty() {
        println!("Clustering resulted in no clusters to analyze.");
        return Ok(());
    }

    println!("Fraud Detection Analysis Results:");

    let mut analyzed_clusters: Vec<(ClusterAnalysis, f64)> = cluster_results.into_iter()
        .map(|cluster| {
            let fraud_rate = cluster.fraud_count as f64 / cluster.size.max(1) as f64;
            (cluster, fraud_rate)
        })
        .collect();

    analyzed_clusters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nHigh-Risk Clusters (Sorted by Fraud Rate):");
    for (i, (cluster, fraud_rate)) in analyzed_clusters.iter().enumerate() {
        print_cluster_analysis(cluster, i + 1, *fraud_rate);
    }

    let metrics = calculate_metrics(&analyzed_clusters);
    println!("Overall Metrics:");
    println!("Total Transactions: {}", metrics.total_transactions);
    println!("Total Fraudulent: {}", metrics.total_fraud);
    println!("Overall Fraud Rate: {:.2}%", metrics.fraud_rate * 100.0);

    Ok(())
}