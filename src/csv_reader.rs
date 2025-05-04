// Module for reading and parsing transaction data from CSV files. Handles data loading and feature vector conversion.
use std::error::Error;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

// Represents a single transaction record with all relevant fields from the CSV data
#[derive(Debug, Deserialize, Clone)]
pub struct Transaction {
    #[serde(rename = "User_ID")]
    pub user_id: u32,
    #[serde(rename = "Transaction_Amount")]
    pub transaction_amount: Option<f64>,
    #[serde(rename = "Transaction_Type")]
    pub transaction_type: String,
    #[serde(rename = "Time_of_Transaction")]
    pub time_of_transaction: Option<f64>,
    #[serde(rename = "Previous_Fraudulent_Transactions")]
    pub previous_fraudulent_transactions: Option<u32>,
    #[serde(rename = "Account_Age")]
    pub account_age: Option<u32>,
    #[serde(rename = "Number_of_Transactions_Last_24H")]
    pub number_of_transactions_last_24h: Option<u32>,
    #[serde(rename = "Payment_Method")]
    pub payment_method: String,
    #[serde(rename = "Fraudulent")]
    pub fraudulent: Option<u32>,
}

impl Transaction {
    // Converts a transaction into a feature vector for clustering
    // Inputs: None (uses self)
    // Outputs: Vector of f64 values representing the transaction's features
    // Key steps:
    // 1. Extract numerical features
    // 2. Handle missing values with defaults
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.transaction_amount.unwrap_or(0.0),
            self.time_of_transaction.unwrap_or(0.0),
            self.previous_fraudulent_transactions.unwrap_or(0) as f64,
            self.account_age.unwrap_or(0) as f64,
            self.number_of_transactions_last_24h.unwrap_or(0) as f64,
        ]
    }
}

// Reads transaction data from a CSV file
// Inputs: Path to the CSV file
// Outputs: Vector of Transaction objects
// Key steps:
// 1. Open and read file
// 2. Parse CSV records
// 3. Convert to Transaction objects
pub fn read_transactions(file_path: &str) -> Result<Vec<Transaction>, Box<dyn Error>> {
    // Open and read file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut transactions = Vec::new();

    // Parse CSV records
    let mut rdr = csv::Reader::from_reader(reader);
    for result in rdr.deserialize() {
        let record: Transaction = result?;
        transactions.push(record);
    }

    Ok(transactions)
} 