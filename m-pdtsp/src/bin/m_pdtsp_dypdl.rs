use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs,
};
use m_pdtsp::{Args, RoundedInstance, SolverChoice};
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

    let instance = Instance::load(&filepath).unwrap();
    let instance = RoundedInstance::try_from(instance).unwrap();

    let mut model = Model::default();

    let n = instance.nodes.len();
    let customer = model.add_object_type("customer", n).unwrap();
    let goal = n - 1;

    let unvisited = (1..goal).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model.add_element_variable("current", customer, 0).unwrap();
    let load = model
        .add_integer_resource_variable("load", true, 0)
        .unwrap();

    let demands = instance
        .demands
        .iter()
        .map(|d| d.iter().sum())
        .collect::<Vec<i32>>();
    let (predecessors, distances) = instance.extract_predecessors_and_filtered_distances();
    let predecessors = predecessors
        .iter()
        .map(|p| {
            let set = p.ones().collect::<Vec<_>>();

            model.create_set(customer, &set).unwrap()
        })
        .collect::<Vec<_>>();
    let connected = distances
        .iter()
        .map(|row| row.iter().map(|&x| x.is_some()).collect())
        .collect();
    let connected = model.add_table_2d("connected", connected).unwrap();
    let min_to = algorithms::take_column_wise_min_with_option(&distances)
        .map(|x| x.unwrap_or(0))
        .collect();
    let min_to = model.add_table_1d("min_to", min_to).unwrap();
    let min_from = algorithms::take_row_wise_min_with_option(&distances)
        .map(|x| x.unwrap_or(0))
        .collect();
    let min_from = model.add_table_1d("min_from", min_from).unwrap();
    let distances = distances
        .iter()
        .map(|d| d.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let distances = model.add_table_2d("distances", distances).unwrap();

    for (next, &d) in demands.iter().enumerate() {
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(distances.element(current, next) + IntegerExpression::Cost);

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        let new_load = load + d;
        visit.add_effect(load, new_load.clone()).unwrap();

        visit.add_precondition(connected.element(current, next));
        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            new_load,
            instance.capacity,
        ));
        visit.add_precondition((unvisited & predecessors[next].clone()).is_empty());

        model.add_forward_transition(visit).unwrap();
    }

    model
        .add_base_case_with_cost(
            vec![connected.element(current, goal), unvisited.is_empty()],
            distances.element(current, goal),
        )
        .unwrap();

    model
        .add_dual_bound(min_to.sum(unvisited) + min_to.element(goal))
        .unwrap();
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
        _ => {
            panic!("Solver {:?} is not supported in dypdl mode. Only Cabs and Astar are available.", args.solver)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let tour = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name())
            .collect::<Vec<_>>();
        println!("Tour: {}", tour.join(" "));
        let tour = tour
            .into_iter()
            .map(|t| t.parse().unwrap())
            .collect::<Vec<_>>();

        if instance.validate(&tour, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
