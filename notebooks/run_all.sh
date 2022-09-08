# Run all the Jupyter notebooks and write the output to disk

run () {
    base=$1
    papermill $base.ipynb output/$base.ipynb
}

mkdir output -p

run notebook_benchmark-toyctf
run notebook_benchmark-chain
run notebook_benchmark-tiny
run notebook_dql_transfer

run c2_interactive_interface
run chainnetwork-optionwrapper
run chainnetwork-random
run randomnetwork
run toyctf-blank
run toyctf-random
run toyctf-solved

run dql_active_directory
run notebook_randlookups
run notebook_tabularq
run notebook_withdefender
run random_active_directory
