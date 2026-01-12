use clap::{Parser, ValueEnum};
use rpid::io;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub profits: Vec<i32>,
    pub weights: Vec<Vec<i32>>,
    pub capacities: Vec<i32>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        let n = digits
            .next()
            .ok_or("failed to parse the number of items".to_owned())?
            .parse::<usize>()?;
        let m = digits
            .next()
            .ok_or("failed to parse the dimension".to_owned())?
            .parse::<usize>()?;
        digits.next();
        let profits = io::read_vector(&mut digits, n)?;
        let weights = io::read_matrix(&mut digits, m, n)?;
        let capacities = io::read_vector(&mut digits, m)?;

        Ok(Self {
            profits,
            weights,
            capacities,
        })
    }

    pub fn validate(&self, solution: &[usize], profit: i32) -> bool {
        let m = self.capacities.len();

        for j in 0..m {
            let total_weight = solution.iter().map(|&i| self.weights[j][i]).sum::<i32>();

            if total_weight > self.capacities[j] {
                println!(
                    "Total weight {total_weight} in dimension {j} exceeds capacity {capacity}",
                    capacity = self.capacities[j]
                );

                return false;
            }
        }

        let recomputed_profit = solution.iter().map(|&i| self.profits[i]).sum::<i32>();

        if recomputed_profit != profit {
            println!("Invalid profit: {profit} != {recomputed_profit}");

            return false;
        }

        true
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    BlindCabs,
    Astar,
    Dijkstra,
    PartialExpansionAstar,
    SmaStar,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(help = "Input file")]
    pub input_file: String,
    #[arg(short, long, value_enum, default_value_t = SolverChoice::Cabs, help = "Solver")]
    pub solver: SolverChoice,
    #[arg(long, default_value_t = String::from("history.csv"), help = "File to save the history")]
    pub history: String,
    #[arg(short, long, default_value_t = 1800.0, help = "Time limit")]
    pub time_limit: f64,
    #[arg(long, default_value_t = 0, help = "Delta parameter for Partial Expansion A* (cutoff value). Delta = 0 stores only nodes with l-value not exceeding optimal cost, Delta = i32::MAX is identical to standard A*")]
    pub pe_delta: i32,
    #[arg(long, default_value_t = 1000000, help = "Maximum queue size for SMA* (number of nodes to keep in the queue)")]
    pub sma_max_queue_size: usize,
    #[arg(
        short,
        long,
        default_value_t = 1e-6,
        help = "Threshold for floating point values"
    )]
    pub epsilon: f64,
}
