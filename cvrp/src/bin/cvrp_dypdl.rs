use clap::Parser;
use cvrp::{Args, RoundedInstance, SolverChoice};
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs,
};
use regex::Regex;
use rpid::{algorithms, timer::Timer};
use std::rc::Rc;
use tsplib_parser::Instance;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let filepath = args.input_file;
    let filename = filepath.split('/').next_back().unwrap();

    let re = Regex::new(r".+k(\d+).+").unwrap();
    let n_vehicles = re.captures(filename).unwrap()[1].parse().unwrap();

    let instance = Instance::load(&filepath).unwrap();
    let mut instance = RoundedInstance::new(instance, n_vehicles).unwrap();
    let n_vehicles = n_vehicles as i32;

    if args.reduce_edges {
        instance.reduce_edges();
    }

    let depot = instance.depot;

    let mut model = Model::default();

    let n = instance.nodes.len();
    let customer = model.add_object_type("customer", n).unwrap();

    let unvisited = (0..n).filter(|&i| i != depot).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model
        .add_element_variable("current", customer, depot)
        .unwrap();
    let load = model
        .add_integer_resource_variable("load", true, 0)
        .unwrap();
    let k = model.add_integer_resource_variable("k", true, 1).unwrap();

    let distances = instance
        .distances
        .iter()
        .map(|row| row.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let distances = model.add_table_2d("distances", distances).unwrap();

    for next in (0..n).filter(|&i| i != depot) {
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(distances.element(current, next) + IntegerExpression::Cost);

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        visit
            .add_effect(load, load + instance.demands[next])
            .unwrap();

        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            load + instance.demands[next],
            instance.capacity,
        ));

        model.add_forward_transition(visit).unwrap();
    }

    let distances_via_depot = instance
        .distances
        .iter()
        .map(|row| {
            {
                (0..n).map(|j| {
                    if let (Some(distance_to_depot), Some(distance_from_depot)) =
                        (row[depot], instance.distances[depot][j])
                    {
                        distance_to_depot + distance_from_depot
                    } else {
                        0
                    }
                })
            }
            .collect()
        })
        .collect();
    let distances_via_depot = model
        .add_table_2d("distances_via_depot", distances_via_depot)
        .unwrap();

    for next in (0..n).filter(|&i| i != depot) {
        let mut visit_via_depot = Transition::new(format!("{}", n + next));
        visit_via_depot
            .set_cost(distances_via_depot.element(current, next) + IntegerExpression::Cost);

        visit_via_depot
            .add_effect(unvisited, unvisited.remove(next))
            .unwrap();
        visit_via_depot.add_effect(current, next).unwrap();
        visit_via_depot
            .add_effect(load, instance.demands[next])
            .unwrap();
        visit_via_depot.add_effect(k, k + 1).unwrap();

        visit_via_depot.add_precondition(unvisited.contains(next));
        visit_via_depot.add_precondition(Condition::comparison_e(
            ComparisonOperator::Ne,
            current,
            depot,
        ));
        visit_via_depot.add_precondition(Condition::comparison_i(
            ComparisonOperator::Lt,
            k,
            n_vehicles,
        ));

        model.add_forward_transition(visit_via_depot).unwrap();
    }

    model
        .add_base_case_with_cost(
            vec![unvisited.is_empty()],
            distances.element(current, depot),
        )
        .unwrap();

    let demands = model
        .add_table_1d("demands", instance.demands.clone())
        .unwrap();
    let total_remaining_capacity = (n_vehicles - k) * instance.capacity + instance.capacity;
    let total_remaining_demand = load + demands.sum(unvisited);
    model
        .add_state_constraint(Condition::comparison_i(
            ComparisonOperator::Ge,
            total_remaining_capacity,
            total_remaining_demand,
        ))
        .unwrap();

    let min_to = algorithms::take_column_wise_min_with_option(&instance.distances)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let min_to = model.add_table_1d("min_to", min_to).unwrap();
    model
        .add_dual_bound(min_to.sum(unvisited) + min_to.element(depot))
        .unwrap();

    let min_from = algorithms::take_row_wise_min_with_option(&instance.distances)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let min_from = model.add_table_1d("min_from", min_from).unwrap();
    model
        .add_dual_bound(min_from.sum(unvisited) + min_from.element(current))
        .unwrap();

    let model = Rc::new(model);

    let parameters = Parameters::<i32> {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let mut solver = match args.solver {
        SolverChoice::Cabs => {
            let beam_search_parameters = BeamSearchParameters {
                parameters,
                ..Default::default()
            };
            let parameters = CabsParameters {
                beam_search_parameters,
                ..Default::default()
            };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            create_dual_bound_cabs(model, parameters, FEvaluatorType::Plus)
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            create_caasdy(model, parameters, FEvaluatorType::Plus)
        }
        _ => panic!(
            "Solver {:?} is not supported in dypdl mode. Only Cabs and Astar are available.",
            args.solver
        ),
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let mut tours = vec![vec![]];

        for transition in solution.transitions {
            let i = transition.get_full_name().parse::<usize>().unwrap();

            if i >= n {
                tours.push(vec![i - n]);
            } else {
                tours.last_mut().unwrap().push(i);
            }
        }

        instance.print_solution(&tours);

        if instance.validate(&tours, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
