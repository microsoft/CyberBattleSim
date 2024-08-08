
#!/usr/bin/python3.9

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Run all notebooks fully to produce publishable benchmark results
# and reusable notebooks

set -ex

kernel=$1
if [ -z "$kernel" ]; then
    kernel=cybersim
fi

script_dir=$(dirname "$0")

pushd "$script_dir/.."

output_dir=notebooks/output/benchmark
output_plot_dir=$output_dir/plots


run () {
    base=$1
    suffix=$2
    cat notebook notebooks/$base.py \
        | jupytext --to ipynb  - \
        | papermill --kernel $kernel $output_dir/$base$suffix.ipynb  "${@:3}"
}

jupyter kernelspec list

mkdir $output_dir -p
mkdir $output_plot_dir -p

# run c2_interactive_interface ''  # disabled: not deterministic and can fail

# run random_active_directory '' # disabled: not deterministic and can fail

# run notebook_dql_debug.py ''


run toyctf-blank ''

run toyctf-random ''

run toyctf-solved ''

run chainnetwork-optionwrapper ''

run chainnetwork-random '' -y "
    iterations: 100
"

run randomnetwork ''

run notebook_benchmark '-chain' -y "
    gymid: 'CyberBattleChain-v0'
    iteration_count: 2000           # iteration_count = 9000
    training_episode_count: 20      # training_episode_count = 50
    eval_episode_count: 3           # eval_episode_count = 5
    maximum_node_count: 20          # maximum_node_count = 22
    maximum_total_credentials: 20   # maximum_total_credentials = 22
    env_size: 10
    plots_dir: $output_plot_dir
"

run notebook_benchmark '-toyctf' -y "
    gymid: 'CyberBattleToyCtf-v0'
    env_size: null
    iteration_count: 1500
    training_episode_count: 20
    eval_episode_count: 10
    maximum_node_count: 12
    maximum_total_credentials: 10
    plots_dir: $output_plot_dir
"

run notebook_benchmark '-tiny' -y "
    gymid: 'CyberBattleTiny-v0'
    env_size: null
    iteration_count: 200
    training_episode_count: 10
    eval_episode_count: 10
    maximum_node_count: 5
    maximum_total_credentials: 3
    plots_dir: $output_plot_dir
"

run notebook_dql_transfer '' -y "
    plots_dir: $output_plot_dir
"

run notebook_randlookups '' -y "
    plots_dir: $output_plot_dir
"

run notebook_tabularq '' -y "
    plots_dir: $output_plot_dir
"

run notebook_withdefender '' -y "
    plots_dir: $output_plot_dir
"

run dql_active_directory '' -y "
  iteration_count: 50
"

popd
