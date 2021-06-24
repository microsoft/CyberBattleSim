# Run all the Jupyter notebooks and write the output to disk

run () {
    base=$1
    papermill $base.ipynb output/$base.ipynb
}

mkdir output -p
run c2_interactive_interface
run chainnetwork-optionwrapper
run chainnetwork-random
run randomnetwork
run toyctf-blank
run toyctf-random
run toyctf-solved
