use clap::Parser;
use fixedbitset::FixedBitSet;
use optw::{Args, Instance, RoundedInstance, SolverChoice};
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp;
use std::cmp::Ordering;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct Optw {
    instance: RoundedInstance,
    shortest_distances: Vec<Vec<i32>>,
    min_distance_from: Vec<i32>,
    min_distance_to: Vec<i32>,
    epsilon: f64,
}

impl Optw {
    fn new(instance: RoundedInstance, epsilon: f64) -> Self {
        let shortest_distances = optw::compute_pairwise_shortest_path_costs(&instance.distances);

        let min_distance_from = algorithms::take_row_wise_min_without_diagonal(&instance.distances)
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();
        let min_distance_to =
            algorithms::take_column_wise_min_without_diagonal(&instance.distances)
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();

        Self {
            instance,
            shortest_distances,
            min_distance_from,
            min_distance_to,
            epsilon,
        }
    }
}

struct OptwState {
    unvisited: FixedBitSet,
    current: usize,
    time: i32,
}

impl Dp for Optw {
    type State = OptwState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let mut unvisited = FixedBitSet::with_capacity(self.instance.vertices.len());
        unvisited.insert_range(1..);

        OptwState {
            unvisited,
            current: 0,
            time: 0,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let n = self.instance.vertices.len();

        for next in state.unvisited.ones() {
            let shortest_time = state.time + self.shortest_distances[state.current][next];

            if shortest_time > self.instance.closing[next]
                || shortest_time + self.shortest_distances[next][0] > self.instance.closing[0]
            {
                let mut unvisited = state.unvisited.clone();
                unvisited.remove(next);
                let state = OptwState {
                    unvisited,
                    current: state.current,
                    time: state.time,
                };

                return vec![(state, 0, n + next)];
            }
        }

        if let Some(next) = state.unvisited.ones().next() {
            if state.unvisited.ones().all(|via| {
                let via_time = state.time + self.instance.distances[state.current][via];

                (via_time > self.instance.closing[via])
                    || (via_time + self.shortest_distances[via][0] > self.instance.closing[0])
            }) {
                let mut unvisited = state.unvisited.clone();
                unvisited.remove(next);
                let state = OptwState {
                    unvisited,
                    current: state.current,
                    time: state.time,
                };

                return vec![(state, 0, 2 * n + next)];
            }
        }

        state
            .unvisited
            .ones()
            .filter_map(|next| {
                let time = state.time + self.instance.distances[state.current][next];

                if time <= self.instance.closing[next]
                    && time + self.shortest_distances[next][0] <= self.instance.closing[0]
                {
                    let mut unvisited = state.unvisited.clone();
                    unvisited.remove(next);
                    let time = cmp::max(time, self.instance.opening[next]);
                    let state = OptwState {
                        unvisited,
                        current: next,
                        time,
                    };

                    Some((state, self.instance.profits[next], next))
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unvisited.is_clear()
            && state.time + self.instance.distances[state.current][0] <= self.instance.closing[0]
        {
            Some(0)
        } else {
            None
        }
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Maximization
    }
}

impl Dominance for Optw {
    type State = OptwState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &Self::State) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(b.time.cmp(&a.time))
    }
}

impl Bound for Optw {
    type State = OptwState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let candidates = state
            .unvisited
            .ones()
            .filter(|&i| {
                let shortest_time = state.time + self.shortest_distances[state.current][i];

                shortest_time <= self.instance.closing[i]
                    && shortest_time + self.shortest_distances[i][0] <= self.instance.closing[0]
            })
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            return Some(0);
        }

        let max_profit_bound = candidates
            .iter()
            .map(|&i| self.instance.profits[i])
            .sum::<i32>();

        let capacity_from =
            self.instance.closing[0] - state.time - self.min_distance_from[state.current];
        let max_efficiency_from = candidates
            .iter()
            .map(|&i| {
                self.instance.profits[i] as f64 / self.min_distance_from[i] as f64 + self.epsilon
            })
            .reduce(|acc, x| if x > acc { x } else { acc })
            .unwrap();
        let efficiency_from_bound = (capacity_from as f64 * max_efficiency_from).floor() as i32;

        let capacity_to = self.instance.closing[0] - state.time - self.min_distance_to[0];
        let max_efficiency_to = candidates
            .into_iter()
            .map(|i| {
                self.instance.profits[i] as f64 / self.min_distance_to[i] as f64 + self.epsilon
            })
            .reduce(|acc, x| if x > acc { x } else { acc })
            .unwrap();
        let efficiency_to_bound = (capacity_to as f64 * max_efficiency_to).floor() as i32;

        Some(cmp::min(
            max_profit_bound,
            cmp::min(efficiency_from_bound, efficiency_to_bound),
        ))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let rounded_instance = RoundedInstance::new(instance, args.round_to);
    let optw = Optw::new(rounded_instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_cabs(optw, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::BlindCabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_blind_cabs(optw, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(optw, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Dijkstra => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_dijkstra(optw, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::PartialExpansionAstar => {
            let pe_parameters = PartialExpansionParameters {
                delta: args.pe_delta,
            };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_partial_expansion_astar(optw, parameters, pe_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::SmaStar => {
            let sma_parameters = SmaStarParameters {
                max_queue_size: args.sma_max_queue_size,
            };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_smastar(optw, parameters, sma_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(profit) = solution.cost {
        let tour = solution
            .transitions
            .into_iter()
            .filter(|&i| i < rounded_instance.vertices.len())
            .collect::<Vec<_>>();
        rounded_instance.print_solution(&tour);

        if rounded_instance.validate(&tour, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
