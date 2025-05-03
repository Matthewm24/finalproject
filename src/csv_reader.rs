use std::error::Error;
use std::fs::File;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Transaction {
    #[serde(rename = "Transaction_ID")]
    pub transaction_id: String,
    #[serde(rename = "User_ID")]
    pub user_id: i32,
    #[serde(rename = "Transaction_Amount")]
    pub transaction_amount: Option<f64>,
    #[serde(rename = "Transaction_Type")]
    pub transaction_type: String,
    #[serde(rename = "Time_of_Transaction")]
    pub time_of_transaction: Option<f64>,
    #[serde(rename = "Device_Used")]
    pub device_used: Option<String>,
    #[serde(rename = "Location")]
    pub location: Option<String>,
    #[serde(rename = "Previous_Fraudulent_Transactions")]
    pub previous_fraudulent_transactions: Option<i32>,
    #[serde(rename = "Account_Age")]
    pub account_age: Option<i32>,
    #[serde(rename = "Number_of_Transactions_Last_24H")]
    pub number_of_transactions_last_24h: Option<i32>,
    #[serde(rename = "Payment_Method")]
    pub payment_method: Option<String>,
    #[serde(rename = "Fraudulent")]
    pub fraudulent: Option<i32>,
}

impl Transaction {
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Amount-based features (handle non-finite values)
        let amount = self.transaction_amount.unwrap_or(0.0);
        features.push(if amount.is_finite() { amount } else { 0.0 });
        features.push(if amount > 0.0 { amount.log10() } else { 0.0 });
        
        // Time-based features (handle non-finite values)
        let time = self.time_of_transaction.unwrap_or(0.0);
        features.push(if time.is_finite() { time } else { 0.0 });
        features.push(if time.is_finite() { (time % 24.0) / 24.0 } else { 0.0 });
        
        // User behavior features
        features.push(self.previous_fraudulent_transactions.unwrap_or(0) as f64);
        features.push(self.account_age.unwrap_or(0) as f64);
        features.push(self.number_of_transactions_last_24h.unwrap_or(0) as f64);
        
        // Transaction type encoding
        let transaction_type = match self.transaction_type.as_str() {
            "ATM Withdrawal" => 0.0,
            "Bill Payment" => 1.0,
            "Online Purchase" => 2.0,
            "Transfer" => 3.0,
            _ => 4.0,
        };
        features.push(transaction_type);
        
        // Payment method encoding
        let payment_method = match self.payment_method.as_deref() {
            Some("Credit Card") => 0.0,
            Some("Debit Card") => 1.0,
            Some("PayPal") => 2.0,
            Some("Bank Transfer") => 3.0,
            _ => 4.0,
        };
        features.push(payment_method);
        
        features
    }
}

pub fn read_transactions(file_path: &str) -> Result<Vec<Transaction>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    
    let transactions: Vec<Transaction> = rdr
        .deserialize()
        .collect::<Result<Vec<Transaction>, csv::Error>>()?;
    
    Ok(transactions)
} 