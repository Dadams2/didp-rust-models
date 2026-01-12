use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs,
};
use graph_clear::{Args, Instance, SolverChoice};
use rpid::timer::Timer;
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

    let mut model = Model::default();

    let n = instance.node_weights.len();
    let node = model
        .add_object_type("node", instance.node_weights.len())
        .unwrap();

    let clean = vec![];
    let clean = model.create_set(node, &clean).unwrap();
    let clean = model.add_set_variable("clean", node, clean).unwrap();

    let edge_weight_sum = instance
        .edge_weights
        .iter()
        .map(|w| w.iter().sum())
        .collect::<Vec<i32>>();
    let edge_weights = model
        .add_table_2d("edge weights", instance.edge_weights.clone())
        .unwrap();
    let contaminated = model
        .add_set_state_function("contaminated", !clean)
        .unwrap();

    for (i, &wi) in instance.node_weights.iter().enumerate() {
        let mut sweep = Transition::new(format!("{i}"));
        sweep.set_cost(IntegerExpression::max(
            wi + edge_weight_sum[i] + edge_weights.sum(clean, contaminated.clone().remove(i)),
            IntegerExpression::Cost,
        ));
        sweep.add_effect(clean, clean.add(i)).unwrap();
        sweep.add_precondition(contaminated.clone().contains(i));

        model.add_forward_transition(sweep).unwrap();
    }

    model
        .add_base_case(vec![Condition::comparison_i(
            ComparisonOperator::Eq,
            clean.len(),
            n as i32,
        )])
        .unwrap();

    model.add_dual_bound(IntegerExpression::from(0)).unwrap();

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

            create_dual_bound_cabs(model, parameters, FEvaluatorType::Max)
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            create_caasdy(model, parameters, FEvaluatorType::Max)
        }
        _ => {
            panic!("Solver {:?} is not supported in dypdl mode. Only Cabs and Astar are available.", args.solver)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let schedule = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name())
            .collect::<Vec<_>>();
        println!("Schedule: {schedule}", schedule = schedule.join(" "));
        let schedule = schedule
            .into_iter()
            .map(|t| t.parse().unwrap())
            .collect::<Vec<usize>>();

        if instance.validate(&schedule, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
