use clap::Parser;
use knapsack::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::cmp::{self, Ordering};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct Knapsack {
    instance: Instance,
    total_profit_after: Vec<i32>,
    max_efficiency_after: Vec<f64>,
}

impl Knapsack {
    fn new(instance: Instance, epsilon: f64) -> Self {
        let mut total_profit_after = instance
            .profits
            .iter()
            .rev()
            .scan(0, |acc, &x| {
                *acc += x;

                Some(*acc)
            })
            .collect::<Vec<_>>();
        total_profit_after.reverse();

        let mut max_efficiency_after = instance
            .profits
            .iter()
            .zip(instance.weights.iter())
            .map(|(&p, &w)| p as f64 / w as f64 + epsilon)
            .rev()
            .scan(0.0, |acc, x| {
                if *acc < x {
                    *acc = x;
                }

                Some(*acc)
            })
            .collect::<Vec<_>>();
        max_efficiency_after.reverse();

        Self {
            instance,
            total_profit_after,
            max_efficiency_after,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct KnapsackState {
    current: usize,
    remaining: i32,
}

impl Dp for Knapsack {
    type State = KnapsackState;
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State {
        KnapsackState {
            current: 0,
            remaining: self.instance.capacity,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let ignore = KnapsackState {
            current: state.current + 1,
            remaining: state.remaining,
        };

        if self.instance.weights[state.current] > state.remaining {
            vec![(ignore, 0, false)]
        } else {
            let pack = KnapsackState {
                current: state.current + 1,
                remaining: state.remaining - self.instance.weights[state.current],
            };

            vec![
                (pack, self.instance.profits[state.current], true),
                (ignore, 0, false),
            ]
        }
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            Some(0)
        } else {
            None
        }
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Maximization
    }
}

impl Dominance for Knapsack {
    type State = KnapsackState;
    type Key = usize;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.current
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(a.remaining.cmp(&b.remaining))
    }
}

impl Bound for Knapsack {
    type State = KnapsackState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            return Some(0);
        }

        let maximum_total_profit = self.total_profit_after[state.current];

        let maximum_efficiency_bound =
            (state.remaining as f64 * self.max_efficiency_after[state.current]).floor() as i32;

        Some(cmp::min(maximum_total_profit, maximum_efficiency_bound))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let knapsack = Knapsack::new(instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_cabs(knapsack, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::BlindCabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_blind_cabs(knapsack, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(knapsack, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Dijkstra => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_dijkstra(knapsack, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::PartialExpansionAstar => {
            let pe_parameters = PartialExpansionParameters { delta: args.pe_delta };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_partial_expansion_astar(knapsack, parameters, pe_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::SmaStar => {
            let sma_parameters = SmaStarParameters { max_queue_size: args.sma_max_queue_size };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_smastar(knapsack, parameters, sma_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(profit) = solution.cost {
        let packed_items = solution
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x { Some(i) } else { None })
            .collect::<Vec<_>>();
        instance.print_solution(&packed_items);

        if instance.validate(&packed_items, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
