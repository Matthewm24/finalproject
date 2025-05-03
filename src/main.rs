use std::error::Error;
use petgraph::graph::{Graph, NodeIndex, UnGraph};
use petgraph::algo::connected_components;
use std::collections::{HashMap, HashSet};
use ndarray::Array;

mod csv_reader;
mod kmeans;

use csv_reader::{read_transactions, Transaction};
use kmeans::{analyze_clusters, ClusterAnalysis};

// Comment out HDBSCAN if no suitable crate is readily available or configured
// use hdbscan::{Hdbscan, HdbscanHyperParams}; // Hypothetical import

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

// Calculate cosine similarity between two feature vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

// Build a graph where nodes are transactions and edges represent high similarity
fn build_transaction_graph(transactions: &[Transaction]) -> UnGraph<&Transaction, f64> {
    let mut graph: UnGraph<&Transaction, f64> = Graph::new_undirected();
    let mut node_indices = HashMap::new();

    // Add nodes for all transactions
    for transaction in transactions {
        let node_index = graph.add_node(transaction);
        node_indices.insert(transaction.transaction_id.clone(), node_index);
    }

    // Convert transactions to feature vectors
    let features: Vec<Vec<f64>> = transactions.iter()
        .map(|tx| tx.to_feature_vector())
        .collect();

    // Calculate similarity and add edges
    let n = transactions.len();
    let similarity_threshold = 0.7;

    for i in 0..n {
        let node1 = node_indices.get(&transactions[i].transaction_id).unwrap();
        for j in (i + 1)..n {
            let similarity = cosine_similarity(&features[i], &features[j]);
            if similarity > similarity_threshold {
                let node2 = node_indices.get(&transactions[j].transaction_id).unwrap();
                graph.add_edge(*node1, *node2, similarity);
            }
        }
    }

    graph
}


// --- Analysis Functions ---

fn calculate_component_density(graph: &UnGraph<&Transaction, f64>, component_nodes: &[NodeIndex]) -> f64 {
    let n = component_nodes.len();
    if n <= 1 {
        return 0.0;
    }

    let max_possible_edges = (n * (n - 1)) / 2;
    if max_possible_edges == 0 {
        return 0.0;
    }

    let mut edge_pairs = HashSet::new();
    let component_set: HashSet<_> = component_nodes.iter().cloned().collect();

    for &node in component_nodes {
        for neighbor in graph.neighbors(node) {
            if component_set.contains(&neighbor) {
                let u = std::cmp::min(node.index(), neighbor.index());
                let v = std::cmp::max(node.index(), neighbor.index());
                edge_pairs.insert((u, v));
            }
        }
    }

    edge_pairs.len() as f64 / max_possible_edges as f64
}


fn calculate_average_degree_centrality(graph: &UnGraph<&Transaction, f64>, component_nodes: &[NodeIndex]) -> f64 {
    let n = component_nodes.len();
    if n <= 1 { // Centrality is ill-defined for isolated nodes in this context
        return 0.0;
    }

    let total_degree: usize = component_nodes.iter()
        .map(|&node| graph.neighbors(node).count())
        .sum();

    // Average degree
    let avg_degree = total_degree as f64 / n as f64;

    // Normalize by the maximum possible degree (n-1)
    avg_degree / (n - 1) as f64
}


fn analyze_fraud_patterns(graph: &UnGraph<&Transaction, f64>, transactions: &[Transaction]) {
    // Find connected components
    let num_components = connected_components(graph);
    println!("\nNumber of connected components: {}", num_components);
    
    let mut fraud_rings = Vec::new();
    let mut visited = HashSet::new();
    let mut user_fraud_counts = HashMap::new();
    let mut user_transaction_counts = HashMap::new();
    
    // Track user behavior
    for tx in transactions {
        *user_transaction_counts.entry(tx.user_id).or_insert(0) += 1;
        if tx.fraudulent == Some(1) {
            *user_fraud_counts.entry(tx.user_id).or_insert(0) += 1;
        }
    }
    
    // Analyze repeat offenders
    let repeat_offenders: Vec<_> = user_fraud_counts
        .iter()
        .filter(|(_, &count)| count > 1)
        .collect();
    
    println!("\nRepeat Offenders Analysis:");
    println!("Found {} users with multiple fraudulent transactions", repeat_offenders.len());
    for (user_id, fraud_count) in repeat_offenders.iter().take(5) {
        let total_txs = user_transaction_counts.get(user_id).unwrap_or(&0);
        println!("User {}: {} fraudulent transactions out of {} total ({:.1}%)",
            user_id, fraud_count, total_txs,
            (**fraud_count as f64 / *total_txs as f64) * 100.0);
    }
    
    // Analyze fraud rings
    for node in graph.node_indices() {
        if !visited.contains(&node) {
            let mut component = Vec::new();
            let mut stack = vec![node];
            visited.insert(node);
            
            while let Some(current) = stack.pop() {
                component.push(current);
                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
            
            if !component.is_empty() {
                let fraud_count = component.iter()
                    .filter(|&&node| graph.node_weight(node).unwrap().fraudulent == Some(1))
                    .count();
                
                if fraud_count > 0 {
                    let density = calculate_component_density(graph, &component);
                    let centrality = calculate_average_degree_centrality(graph, &component);
                    
                    // Analyze user patterns in the component
                    let mut users_in_component = HashSet::new();
                    let mut user_transactions = HashMap::new();
                    
                    for &node in &component {
                        let tx = graph.node_weight(node).unwrap();
                        users_in_component.insert(tx.user_id);
                        *user_transactions.entry(tx.user_id).or_insert(0) += 1;
                    }
                    
                    fraud_rings.push((component, density, centrality, fraud_count, users_in_component.len(), user_transactions));
                }
            }
        }
    }
    
    // Print detailed analysis
    println!("\nFraud Pattern Analysis:");
    for (i, (component, density, centrality, fraud_count, user_count, user_txs)) in fraud_rings.iter().enumerate() {
        println!("\nFraud Ring {}:", i + 1);
        println!("Size: {} transactions", component.len());
        println!("Unique users: {}", user_count);
        println!("Fraudulent transactions: {}", fraud_count);
        println!("Density: {:.2}", density);
        println!("Average Centrality: {:.2}", centrality);
        
        // Print user patterns
        println!("\nUser Activity in Ring:");
        for (user_id, tx_count) in user_txs.iter().take(5) {
            let fraud_count = user_fraud_counts.get(user_id).unwrap_or(&0);
            println!("User {}: {} transactions ({} fraudulent)", 
                user_id, tx_count, fraud_count);
        }
    }
}


fn main() -> Result<(), Box<dyn Error>> {
    let csv_file_path = "fraud_detection.csv";
    println!("Loading data from '{}'...", csv_file_path);
    
    let transactions = read_transactions(csv_file_path)?;
    println!("Loaded {} transactions", transactions.len());
    
    // Run k-means clustering
    println!("\nRunning k-means clustering...");
    let n_clusters = 5;
    let cluster_results = analyze_clusters(&transactions, n_clusters)?;
    
    // Print cluster analysis
    println!("\nK-means Clustering Results:");
    for (i, cluster) in cluster_results.iter().enumerate() {
        println!("\nCluster {}:", i);
        println!("Size: {} transactions", cluster.size);
        println!("Fraudulent transactions: {} ({:.1}%)", 
            cluster.fraud_count, 
            (cluster.fraud_count as f64 / cluster.size as f64) * 100.0);
        println!("Unique users: {}", cluster.unique_users);
        
        println!("\nAverage Feature Values:");
        println!("1. Transaction Amount: {:.2}", cluster.avg_features[0]);
        println!("2. Log Amount: {:.2}", cluster.avg_features[1]);
        println!("3. Transaction Time: {:.2}", cluster.avg_features[2]);
        println!("4. Hour of Day: {:.2}", cluster.avg_features[3]);
        println!("5. Previous Fraud Count: {:.2}", cluster.avg_features[4]);
        println!("6. Account Age: {:.2}", cluster.avg_features[5]);
        println!("7. Transactions Last 24H: {:.2}", cluster.avg_features[6]);
        println!("8. Transaction Type: {:.2}", cluster.avg_features[7]);
        println!("9. Payment Method: {:.2}", cluster.avg_features[8]);
        
        println!("\nFeature Interpretations:");
        println!("Most Common Transaction Type: {}", cluster.most_common_tx_type);
        println!("Most Common Payment Method: {}", cluster.most_common_payment);
    }
    
    // Build and analyze transaction graph
    let transactions_slice: &[Transaction] = &transactions;
    let graph = build_transaction_graph(transactions_slice);
    analyze_fraud_patterns(&graph, transactions_slice);
    
    Ok(())
}