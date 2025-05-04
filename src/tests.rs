use crate::csv_reader::Transaction;
use crate::kmeans::analyze_clusters;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transactions() -> Vec<Transaction> {
        vec![
            Transaction {
                user_id: 1,
                transaction_amount: Some(100.0),
                transaction_type: "Online Purchase".to_string(),
                time_of_transaction: Some(1.0),
                previous_fraudulent_transactions: Some(0),
                account_age: Some(365),
                number_of_transactions_last_24h: Some(2),
                payment_method: "Credit Card".to_string(),
                fraudulent: Some(0),
            },
            Transaction {
                user_id: 2,
                transaction_amount: Some(1000.0),
                transaction_type: "Transfer".to_string(),
                time_of_transaction: Some(2.0),
                previous_fraudulent_transactions: Some(1),
                account_age: Some(30),
                number_of_transactions_last_24h: Some(5),
                payment_method: "Bank Transfer".to_string(),
                fraudulent: Some(1),
            },
            Transaction {
                user_id: 1,
                transaction_amount: Some(50.0),
                transaction_type: "ATM Withdrawal".to_string(),
                time_of_transaction: Some(3.0),
                previous_fraudulent_transactions: Some(0),
                account_age: Some(365),
                number_of_transactions_last_24h: Some(3),
                payment_method: "Debit Card".to_string(),
                fraudulent: Some(0),
            },
        ]
    }

    #[test]
    fn test_transaction_feature_vector() {
        let transaction = create_test_transactions()[0].clone();
        let features = transaction.to_feature_vector();
        assert_eq!(features.len(), 5, "Feature vector should have 5 elements");

        assert_eq!(features[0], 100.0, "First feature should be transaction amount");
        assert_eq!(features[1], 1.0, "Second feature should be time of transaction");
        assert_eq!(features[2], 0.0, "Third feature should be previous fraudulent transactions");
        assert_eq!(features[3], 365.0, "Fourth feature should be account age");
        assert_eq!(features[4], 2.0, "Fifth feature should be number of transactions in last 24h");
    }

    #[test]
    fn test_transaction_feature_vector_with_none_values() {
        let mut transaction = create_test_transactions()[0].clone();
        transaction.transaction_amount = None;
        transaction.time_of_transaction = None;
        transaction.previous_fraudulent_transactions = None;
        transaction.account_age = None;
        transaction.number_of_transactions_last_24h = None;

        let features = transaction.to_feature_vector();
        assert_eq!(features.len(), 5, "Feature vector should still have 5 elements");
        assert_eq!(features[0], 0.0, "None transaction amount should default to 0.0");
        assert_eq!(features[1], 0.0, "None time should default to 0.0");
        assert_eq!(features[2], 0.0, "None previous fraud should default to 0.0");
        assert_eq!(features[3], 0.0, "None account age should default to 0.0");
        assert_eq!(features[4], 0.0, "None transaction count should default to 0.0");
    }

    #[test]
    fn test_clustering_basic() {
        let transactions = create_test_transactions();
        let n_clusters = 2;
        let result = analyze_clusters(&transactions, n_clusters);

        assert!(result.is_ok(), "Clustering should succeed");
        let clusters = result.unwrap();
        assert!(clusters.len() <= n_clusters, "Should not have more clusters than requested");
        assert!(!clusters.is_empty(), "Should have at least one cluster");
    }

    #[test]
    fn test_cluster_analysis_properties() {
        let transactions = create_test_transactions();
        let n_clusters = 2;
        let clusters = analyze_clusters(&transactions, n_clusters).unwrap();

        for cluster in clusters {
            // Test size properties
            assert!(cluster.size > 0, "Cluster size should be positive");
            assert!(cluster.fraud_count <= cluster.size, "Fraud count cannot exceed cluster size");
            assert!(cluster.unique_users <= cluster.size, "Unique users cannot exceed cluster size");

            assert_eq!(cluster.avg_features.len(), 5, "Should have 5 average features");
            for feature in cluster.avg_features {
                assert!(!feature.is_nan(), "Average features should not be NaN");
                assert!(!feature.is_infinite(), "Average features should not be infinite");
            }
            assert!(!cluster.most_common_tx_type.is_empty(), "Transaction type should not be empty");
            assert!(!cluster.most_common_payment.is_empty(), "Payment method should not be empty");
        }
    }

    #[test]
    fn test_empty_transactions() {
        let empty_transactions: Vec<Transaction> = vec![];
        let result = analyze_clusters(&empty_transactions, 2);
        
        assert!(result.is_ok(), "Should handle empty transactions gracefully");
        assert!(result.unwrap().is_empty(), "Should return empty clusters for empty transactions");
    }

    #[test]
    fn test_single_transaction() {
        let single_transaction = vec![create_test_transactions()[0].clone()];
        let result = analyze_clusters(&single_transaction, 2);
        
        assert!(result.is_ok(), "Should handle single transaction");
        let clusters = result.unwrap();
        assert!(!clusters.is_empty(), "Should create at least one cluster");
        
        if let Some(cluster) = clusters.first() {
            assert_eq!(cluster.size, 1, "Cluster should contain exactly one transaction");
            assert!(cluster.unique_users == 1, "Should have exactly one unique user");
        }
    }
}
