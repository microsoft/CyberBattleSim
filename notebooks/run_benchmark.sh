# Run benchmarking notebooks

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
    papermill --kernel $kernel notebooks/$base.ipynb $output_dir/$base.ipynb  "${@:2}"
}

jupyter kernelspec list

mkdir $output_dir -p
mkdir $output_plot_dir -p

run notebook_benchmark-chain -y "
    gymid: "CyberBattleChain-v0"
    iteration_count: 2000
    training_episode_count: 20
    eval_episode_count: 3
    maximum_node_count: 20
    maximum_total_credentials: 20
    env_size: 14
    plots_dir: $output_plot_dir
"


popd
