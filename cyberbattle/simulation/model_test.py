# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unit tests for model.py.

Note that model.py mainly provides the data modelling for the simulation,
that is naked data types without members. There is therefore not much
relevant unit testing that can be implemented at this stage.
Once we add operations to generate and modify environments there
will be more room for unit-testing.

"""
# pylint: disable=missing-function-docstring

from cyberbattle.simulation.model import AdminEscalation, Identifiers, SystemEscalation
import yaml
from datetime import datetime

import networkx as nx

from . import model

ADMINTAG = AdminEscalation().tag
SYSTEMTAG = SystemEscalation().tag

vulnerabilities = {
    "UACME61":
        model.VulnerabilityInfo(
            description="UACME UAC bypass #61",
            type=model.VulnerabilityType.LOCAL,
            URL="https://github.com/hfiref0x/UACME",
            precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
            outcome=model.AdminEscalation(),
            rates=model.Rates(0, 0.2, 1.0))}

ENV_IDENTIFIERS = Identifiers(
    properties=[],
    ports=[],
    local_vulnerabilities=["UACME61"],
    remote_vulnerabilities=[]
)


# Verify that there is a unique node injected with the
# attacker in a randomly generated graph
def test_single_infected_node_initially() -> None:
    # create a random environment
    graph = nx.cubical_graph()
    graph = model.assign_random_labels(graph)
    env = model.Environment(network=graph,
                            vulnerability_library=dict([]),
                            identifiers=ENV_IDENTIFIERS)
    count = sum(1 for i in graph.nodes
                if env.get_node(i).agent_installed)
    assert count == 1
    return


# ensures that an environment can successfully be serialized as yaml
def test_environment_is_serializable() -> None:
    # create a random environment
    env = model.Environment(
        network=model.assign_random_labels(nx.cubical_graph()),
        version=model.VERSION_TAG,
        vulnerability_library=dict([]),
        identifiers=ENV_IDENTIFIERS,
        creationTime=datetime.utcnow(),
        lastModified=datetime.utcnow(),
    )
    # Dump the environment as Yaml
    _ = yaml.dump(env)
    assert True


# Test random graph get_node_information
def test_create_random_environment() -> None:
    graph = nx.cubical_graph()

    graph = model.assign_random_labels(graph)

    env = model.Environment(
        network=graph,
        vulnerability_library=vulnerabilities,
        identifiers=ENV_IDENTIFIERS
    )
    assert env
    pass


def check_reserializing(object_to_serialize: object) -> None:
    """Helper function to check that deserializing and serializing are inverse of each other"""
    serialized = yaml.dump(object_to_serialize)
    # print('Serialized: ' + serialized)
    deserialized = yaml.load(serialized, yaml.Loader)
    re_serialized = yaml.dump(deserialized)
    assert (serialized == re_serialized)


def test_yaml_serialization_networkx() -> None:
    """Test Yaml serialization and deserialization for type Environment"""
    model.setup_yaml_serializer()
    check_reserializing(model.assign_random_labels(nx.cubical_graph()))


def test_yaml_serialization_environment() -> None:
    """Test Yaml serialization and deserialization for type Environment

    Note: if `model.Environment` is declared as a NamedTuple instead of a @dataclass
    then this test breaks with networkx>=2.8.1 (but works with 2.8.0)
    due to the new networkx field `edges._graph` self referencing the graph.
    """
    network = model.assign_random_labels(nx.cubical_graph())
    env = model.Environment(
        network=network,
        vulnerability_library=vulnerabilities,
        identifiers=model.infer_constants_from_network(network, vulnerabilities))

    model.setup_yaml_serializer()

    serialized = yaml.dump(env)
    assert (len(serialized) > 100)

    check_reserializing(env)


def test_yaml_serialization_precondition() -> None:
    """Test Yaml serialization and deserialization for type Precondition"""
    model.setup_yaml_serializer()

    precondition = model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))")
    check_reserializing(precondition)

    deserialized = yaml.safe_load(yaml.dump(precondition))
    assert (precondition.expression == deserialized.expression)


def test_yaml_serialization_vulnerabilitytype() -> None:
    """Test Yaml serialization and deserialization for type VulnerabilityType"""
    model.setup_yaml_serializer()

    object_to_serialize = model.VulnerabilityType.LOCAL
    check_reserializing(object_to_serialize)
