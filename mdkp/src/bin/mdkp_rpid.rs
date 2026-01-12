use clap::Parser;
use mdkp::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::cmp;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct Mdkp {
    instance: Instance,
    total_profit_after: Vec<i32>,
    max_efficiencies_after: Vec<Vec<f64>>,
}

impl Mdkp {
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

        let max_efficiencies_after = instance
            .weights
            .iter()
            .map(|ws| {
                let mut ms = instance
                    .profits
                    .iter()
                    .zip(ws)
                    .enumerate()
                    .map(|(i, (&p, &w))| {
                        if w > 0 {
                            p as f64 / w as f64 + epsilon
                        } else {
                            total_profit_after[i] as f64
                        }
                    })
                    .rev()
                    .scan(0.0, |acc, x| {
                        if *acc < x {
                            *acc = x;
                        }

                        Some(*acc)
                    })
                    .collect::<Vec<_>>();
                ms.reverse();

                ms
            })
            .collect();

        Self {
            instance,
            total_profit_after,
            max_efficiencies_after,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MdkpState {
    current: usize,
    remaining: Vec<i32>,
}

impl Dp for Mdkp {
    type State = MdkpState;
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State {
        MdkpState {
            current: 0,
            remaining: self.instance.capacities.clone(),
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let mut remaining = state.remaining.clone();

        for (j, (&r, ws)) in state
            .remaining
            .iter()
            .zip(self.instance.weights.iter())
            .enumerate()
        {
            if ws[state.current] > r {
                remaining[..j]
                    .iter_mut()
                    .zip(state.remaining[..j].iter())
                    .for_each(|(x, y)| *x = *y);
                let successor = MdkpState {
                    current: state.current + 1,
                    remaining,
                };

                return vec![(successor, 0, false)];
            }

            remaining[j] -= ws[state.current];
        }

        let successor_1 = MdkpState {
            current: state.current + 1,
            remaining,
        };
        let successor_2 = MdkpState {
            current: state.current + 1,
            remaining: state.remaining.clone(),
        };

        vec![
            (successor_1, self.instance.profits[state.current], true),
            (successor_2, 0, false),
        ]
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

impl Dominance for Mdkp {
    type State = MdkpState;
    type Key = MdkpState;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for Mdkp {
    type State = MdkpState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            return Some(0);
        }

        let maximum_total_profit = self.total_profit_after[state.current];

        let maximum_efficiency_bound = state
            .remaining
            .iter()
            .zip(self.max_efficiencies_after.iter())
            .map(|(&r, ms)| (cmp::max(r, 1) as f64 * ms[state.current]).floor() as i32)
            .min()
            .unwrap();

        Some(cmp::min(maximum_total_profit, maximum_efficiency_bound))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let mdkp = Mdkp::new(instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_cabs(mdkp, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::BlindCabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_blind_cabs(mdkp, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(mdkp, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Dijkstra => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_dijkstra(mdkp, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::PartialExpansionAstar => {
            let pe_parameters = PartialExpansionParameters { delta: args.pe_delta };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_partial_expansion_astar(mdkp, parameters, pe_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::SmaStar => {
            let sma_parameters = SmaStarParameters { max_queue_size: args.sma_max_queue_size };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_smastar(mdkp, parameters, sma_parameters);
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
        println!(
            "Packed items: {}",
            packed_items
                .iter()
                .map(|&i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );

        if instance.validate(&packed_items, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
