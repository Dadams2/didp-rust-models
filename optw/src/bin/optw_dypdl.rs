use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs,
};
use optw::{Args, Instance, RoundedInstance, SolverChoice};
use rpid::{algorithms, timer::Timer};
use std::rc::Rc;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let rounded_instance = RoundedInstance::new(instance, args.round_to);

    let mut model = Model::default();
    model.set_maximize();

    let n = rounded_instance.vertices.len();
    let customer = model.add_object_type("customer", n).unwrap();

    let unvisited = (1..n).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model.add_element_variable("current", customer, 0).unwrap();
    let time = model
        .add_integer_resource_variable("time", true, 0)
        .unwrap();

    let shortest_distances =
        optw::compute_pairwise_shortest_path_costs(&rounded_instance.distances);
    let shortest_return_distances = shortest_distances
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &d)| d + shortest_distances[j][0])
                .collect()
        })
        .collect();
    let distances_plus_shortest_return = rounded_instance
        .distances
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &d)| d + shortest_distances[j][0])
                .collect()
        })
        .collect();
    let distances = model
        .add_table_2d("distances", rounded_instance.distances.clone())
        .unwrap();
    let shortest_distances = model
        .add_table_2d("shortest_distances", shortest_distances)
        .unwrap();
    let shortest_return_distances = model
        .add_table_2d("shortest_return_distances", shortest_return_distances)
        .unwrap();
    let distances_plus_shortest_return = model
        .add_table_2d(
            "distances_plus_shortest_return",
            distances_plus_shortest_return,
        )
        .unwrap();

    for next in 1..n {
        let mut remove = Transition::new(format!("{}", n + next));
        remove.set_cost(IntegerExpression::Cost);
        remove
            .add_effect(unvisited, unvisited.remove(next))
            .unwrap();
        remove.add_precondition(unvisited.contains(next));

        let shortest_time = time + shortest_distances.element(current, next);
        let shortest_return_time = time + shortest_return_distances.element(current, next);
        remove.add_precondition(
            Condition::comparison_i(
                ComparisonOperator::Gt,
                shortest_time,
                rounded_instance.closing[next],
            ) | Condition::comparison_i(
                ComparisonOperator::Gt,
                shortest_return_time,
                rounded_instance.closing[0],
            ),
        );

        model.add_forward_forced_transition(remove).unwrap();
    }

    for next in 1..n {
        let mut clear = Transition::new(format!("{}", 2 * n + next));
        clear.set_cost(IntegerExpression::Cost);
        clear.add_effect(unvisited, unvisited.remove(next)).unwrap();
        clear.add_precondition(unvisited.contains(next));

        for via in 1..n {
            let via_time = time + distances.element(current, via);
            let shortest_return_time = time + shortest_return_distances.element(current, via);
            clear.add_precondition(
                !unvisited.contains(via)
                    | Condition::comparison_i(
                        ComparisonOperator::Gt,
                        via_time,
                        rounded_instance.closing[via],
                    )
                    | Condition::comparison_i(
                        ComparisonOperator::Gt,
                        shortest_return_time,
                        rounded_instance.closing[0],
                    ),
            );
        }

        model.add_forward_forced_transition(clear).unwrap();
    }

    for next in 1..n {
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(rounded_instance.profits[next] + IntegerExpression::Cost);

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        let arrival_time = time + distances.element(current, next);
        let start_time =
            IntegerExpression::max(arrival_time.clone(), rounded_instance.opening[next]);
        visit.add_effect(time, start_time).unwrap();

        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            arrival_time,
            rounded_instance.closing[next],
        ));
        let shortest_return_time = time + distances_plus_shortest_return.element(current, next);
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_return_time,
            rounded_instance.closing[0],
        ));

        model.add_forward_transition(visit).unwrap();
    }

    let on_time = Condition::comparison_i(
        ComparisonOperator::Le,
        time + distances.element(current, 0),
        rounded_instance.closing[0],
    );
    model
        .add_base_case(vec![unvisited.is_empty(), on_time])
        .unwrap();

    let mut max_profit_bound: Option<IntegerExpression> = None;

    for (i, &p) in rounded_instance.profits.iter().enumerate().skip(1) {
        let shortest_time = time + shortest_distances.element(current, i);
        let on_time = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_time,
            rounded_instance.closing[i],
        );
        let shortest_return_time = time + shortest_return_distances.element(current, i);
        let on_time_return = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_return_time,
            rounded_instance.closing[0],
        );
        let p = (unvisited.contains(i) & on_time & on_time_return).if_then_else(p, 0);

        if max_profit_bound.is_none() {
            max_profit_bound = Some(p);
        } else {
            max_profit_bound = Some(max_profit_bound.unwrap() + p);
        }
    }

    model.add_dual_bound(max_profit_bound.unwrap()).unwrap();

    let min_from = algorithms::take_row_wise_min_without_diagonal(&rounded_instance.distances)
        .map(|x| x.unwrap_or(0))
        .collect::<Vec<_>>();
    let efficiency_from = min_from
        .iter()
        .enumerate()
        .map(|(i, &d)| rounded_instance.profits[i] as f64 / d as f64 + args.epsilon);
    let mut max_efficiency_from = None;

    for (i, e) in efficiency_from.enumerate().skip(1) {
        let shortest_time = time + shortest_distances.element(current, i);
        let on_time = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_time,
            rounded_instance.closing[i],
        );
        let shortest_return_time = time + shortest_return_distances.element(current, i);
        let on_time_return = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_return_time,
            rounded_instance.closing[0],
        );
        let e = (unvisited.contains(i) & on_time & on_time_return).if_then_else(e, 0);

        if max_efficiency_from.is_none() {
            max_efficiency_from = Some(e);
        } else {
            max_efficiency_from = Some(ContinuousExpression::max(max_efficiency_from.unwrap(), e));
        }
    }

    let min_from = model.add_table_1d("min_from", min_from).unwrap();
    let capacity_from = rounded_instance.closing[0] - time - min_from.element(current);
    model
        .add_dual_bound(IntegerExpression::floor(
            capacity_from * max_efficiency_from.unwrap(),
        ))
        .unwrap();

    let min_to = algorithms::take_column_wise_min_without_diagonal(&rounded_instance.distances)
        .map(|x| x.unwrap_or(0))
        .collect::<Vec<_>>();
    let efficiency_to = min_to
        .iter()
        .enumerate()
        .map(|(i, &d)| rounded_instance.profits[i] as f64 / d as f64 + args.epsilon);
    let mut max_efficiency_to = None;

    for (i, e) in efficiency_to.enumerate().skip(1) {
        let shortest_time = time + shortest_distances.element(current, i);
        let on_time = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_time,
            rounded_instance.closing[i],
        );
        let shortest_return_time = time + shortest_return_distances.element(current, i);
        let on_time_return = Condition::comparison_i(
            ComparisonOperator::Le,
            shortest_return_time,
            rounded_instance.closing[0],
        );
        let e = (unvisited.contains(i) & on_time & on_time_return).if_then_else(e, 0);

        if max_efficiency_to.is_none() {
            max_efficiency_to = Some(e);
        } else {
            max_efficiency_to = Some(ContinuousExpression::max(max_efficiency_to.unwrap(), e));
        }
    }

    let capacity_to = rounded_instance.closing[0] - time - min_to[0];
    model
        .add_dual_bound(IntegerExpression::floor(
            capacity_to * max_efficiency_to.unwrap(),
        ))
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

    if let Some(profit) = solution.cost {
        let tour = solution
            .transitions
            .iter()
            .filter_map(|t| {
                let i = t.get_full_name().parse().unwrap();

                if i < n { Some(i) } else { None }
            })
            .collect::<Vec<_>>();
        rounded_instance.print_solution(&tour);

        if rounded_instance.validate(&tour, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
