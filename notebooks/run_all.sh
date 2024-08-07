# Run all the Jupyter notebooks and write the output to disk

set -ex

kernel=$1
if [ -z "$kernel" ]; then
    kernel=cybersim
fi

script_dir=$(dirname "$0")

pushd "$script_dir/.."

run () {
    base=$1
    papermill --kernel $kernel notebooks/$base.ipynb notebooks/output/$base.ipynb  "${@:2}"
}

jupyter kernelspec list

mkdir notebooks/output -p

# run c2_interactive_interface # disabled: not deterministic and can fail

# run  random_active_directory # disabled: not deterministic and can fail

run toyctf-blank

run toyctf-random

run toyctf-solved

run notebook_benchmark-toyctf -y "
    iteration_count: 100
    training_episode_count: 3
    eval_episode_count: 5
    maximum_node_count: 12
    maximum_total_credentials: 10
"

run notebook_benchmark-chain -y "
    iteration_count: 100
    training_episode_count: 5
    eval_episode_count: 3
    maximum_node_count: 12
    maximum_total_credentials: 7
"

run notebook_benchmark-tiny -y "
  iteration_count: 100
  training_episode_count: 4
  eval_episode_count: 2
  maximum_node_count: 5
  maximum_total_credentials: 3
  plots_dir: notebooks/output/plots
"

run notebook_dql_transfer -y "
    iteration_count: 500
    training_episode_count: 5
    eval_episode_count: 3
"

run chainnetwork-optionwrapper

run chainnetwork-random -y "
    iterations: 100
"

run randomnetwork

run notebook_randlookups -y "
    iteration_count: 500
    training_episode_count: 5
    eval_episode_count: 2
"""

run notebook_tabularq -y "
    iteration_count: 200
    training_episode_count: 5
    eval_episode_count: 2
"

run notebook_withdefender -y "
    iteration_count: 100
    training_episode_count: 3
    plots_dir: notebooks/output/plots
"

run dql_active_directory -y "
    ngyms: 3
    iteration_count: 50
"

popd
