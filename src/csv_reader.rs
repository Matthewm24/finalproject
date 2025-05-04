use std::error::Error;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

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

pub fn read_transactions(file_path: &str) -> Result<Vec<Transaction>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut transactions = Vec::new();

    let mut rdr = csv::Reader::from_reader(reader);
    for result in rdr.deserialize() {
        let record: Transaction = result?;
        transactions.push(record);
    }

    Ok(transactions)
} 