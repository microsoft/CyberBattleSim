# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A simple test sandbox to play with creation of simulation environments"""
import networkx as nx
import yaml

from cyberbattle.simulation import model, model_test, actions_test


def main() -> None:
    """Simple environment sandbox"""

    # Create a toy graph
    graph = nx.DiGraph()
    graph.add_edges_from([('a', 'b'), ('b', 'c')])
    print(graph)

    # create a random graph
    graph = nx.cubical_graph()

    graph = model.assign_random_labels(graph)

    vulnerabilities = actions_test.SAMPLE_VULNERABILITIES

    model.setup_yaml_serializer()

    # Define an environment from this graph
    env = model.Environment(
        network=graph,
        vulnerability_library=vulnerabilities,
        identifiers=actions_test.ENV_IDENTIFIERS
    )

    model_test.check_reserializing(env)

    model_test.check_reserializing(vulnerabilities)

    # Save the environment to file as Yaml
    with open('./simpleenv.yaml', 'w') as file:
        yaml.dump(env, file)
    print(yaml.dump(env))


if __name__ == '__main__':
    main()
