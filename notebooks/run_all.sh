# Run all the Jupyter notebooks and write the output to disk

set -ex

kernel=$1
if [ -z "$kernel" ]; then
    kernel=cybersim
fi

script_dir=$(dirname "$0")

# Change the current directory to the parent directory of the script
pushd "$script_dir/.."

run () {
    base=$1
    papermill --kernel $kernel notebooks/$base.ipynb notebooks/output/$base.ipynb
}

jupyter kernelspec list

mkdir notebooks/output -p

declare -a all_notebooks=(
    notebook_benchmark-toyctf
    notebook_benchmark-chain
    notebook_benchmark-tiny
    notebook_dql_transfer
    chainnetwork-optionwrapper
    chainnetwork-random
    randomnetwork
    toyctf-blank
    toyctf-random
    toyctf-solved
    notebook_randlookups
    notebook_tabularq
    notebook_withdefender
)

declare -a excluded=(
    # too long to run
    dql_active_directory
    # not deterministic, can fail
    c2_interactive_interface
    # not deterministic, can fail
    random_active_directory
)

for file in ${all_notebooks[@]}
do
    run $file
done

popd
