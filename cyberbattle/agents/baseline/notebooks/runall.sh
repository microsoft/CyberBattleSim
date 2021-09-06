
#!/usr/bin/python3.8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Run all jupytext notebook and save output under output directory

# notebook_all_agents_benchmark
# notebook_dql_bench
# notebook_dql_debug
# notebook_dql_transfer
# notebook_tabularq
# notebook_randlookups
# notebook_withdefender

cat notebook_dql_debug.py | jupytext --to ipynb --set-kernel - | papermill output/notebook_dql_debug_tiny.ipynb -p gymid 'CyberBattleTiny-v0' -p iteration_count 150

cat notebook_ctf_dql.py | jupytext --to ipynb --set-kernel - | papermill output/notebook_ctf_dql.ipynb -p gymid 'CyberBattleToyCtf-v0' -p iteration_count 1500

cat notebook_dql.py | jupytext --to ipynb --set-kernel - | papermill output/notebook_dql.ipynb

cat notebook_all_agents_benchmark.py | jupytext --to ipynb --set-kernel - | papermill output/notebook_all_agents_benchmark.ipynb
