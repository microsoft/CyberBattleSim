# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
    The unit tests for the environment_generation functions
"""
from collections import Counter
from cyberbattle.simulation import commandcontrol
from typing import List, Dict
import pytest
from . import environment_generation
from . import model

windows_vulns: Dict[str, model.VulnerabilityInfo] = environment_generation.potential_windows_vulns
linux_vulns: Dict[str, model.VulnerabilityInfo] = environment_generation.potential_linux_vulns

windows_node_states: List[model.PropertyName] = environment_generation.potential_linux_node_states
linux_node_states: List[model.PropertyName] = environment_generation.potential_linux_node_states

potential_ports: List[model.PortName] = environment_generation.potential_ports


def test_create_random_environment() -> None:
    """
        The unit tests for create_random_environment function
    """
    with pytest.raises(ValueError, match=r"Please supply a non empty string for the name"):
        environment_generation.create_random_environment("", 2)

    with pytest.raises(ValueError, match=r"Please supply a positive non zero positive"
                                         r"integer for the size of the environment"):
        environment_generation.create_random_environment("Test_environment", -5)

    result: model.Environment = environment_generation.\
        create_random_environment("Test_environment 2", 4)
    assert isinstance(result, model.Environment)


def test_random_environment_list_attacks() -> None:
    """
        Unit tests for #23 caused by bug https://github.com/bastikr/boolean.py/issues/82 in boolean.py
    """
    env = environment_generation.create_random_environment('test', 10)
    c2 = commandcontrol.CommandControl(env)
    c2.print_all_attacks()


def test_create_random_node() -> None:
    """
        The unit tests for create_random_node() function
    """

    # check that the correct exceptions are generated
    with pytest.raises(ValueError, match=r"No endpoints supplied"):
        environment_generation.create_random_node("Linux", [])

    with pytest.raises(ValueError, match=r"Unsupported OS Type please enter Linux or Windows"):
        environment_generation.create_random_node("Solaris", potential_ports)

    test_node: model.NodeInfo = environment_generation.create_random_node("Linux", potential_ports)

    assert isinstance(test_node, model.NodeInfo)


def test_get_properties_from_vulnerabilities() -> None:
    """
        This function tests the get_properties_from_vulnerabilities function
        It takes nothing and returns nothing.
    """
    # testing on linux vulns
    props: List[model.PropertyName] = environment_generation.\
        get_properties_from_vulnerabilities("Linux", linux_vulns)
    assert "Linux" in props
    assert "PortSSHOpen" in props
    assert "PortSMBOpen" in props

    # testing on Windows vulns
    windows_props: List[model.PropertyName] = environment_generation.get_properties_from_vulnerabilities(
        "Windows", windows_vulns)
    assert "Windows" in windows_props
    assert "PortRDPOpen" in windows_props
    assert "PortSMBOpen" in windows_props
    assert "DomainJoined" in windows_props
    assert "Win10" in windows_props
    assert "Win7" in windows_props


def test_create_firewall_rules() -> None:
    """
        This function tests the create_firewall_rules function.
        It takes nothing and returns nothing.
    """
    empty_ports: List[model.PortName] = []
    potential_port_list: List[model.PortName] = ["RDP", "SSH", "HTTP", "HTTPs",
                                                 "SMB", "SQL", "FTP", "WMI"]
    half_ports: List[model.PortName] = ["SSH", "HTTPs", "SQL", "FTP", "WMI"]
    all_blocked: List[model.FirewallRule] = [model.FirewallRule(
        port, model.RulePermission.BLOCK) for port in potential_port_list]
    all_allowed: List[model.FirewallRule] = [model.FirewallRule(
        port, model.RulePermission.ALLOW) for port in potential_port_list]
    half_allowed: List[model.FirewallRule] = [model.FirewallRule(port, model.RulePermission.ALLOW)
                                              if port in half_ports else model.FirewallRule(
                                                  port, model.RulePermission.BLOCK) for
                                              port in potential_port_list]

    # testing on an empty list should lead to
    results: model.FirewallConfiguration = environment_generation.create_firewall_rules(empty_ports)
    assert Counter(results.incoming) == Counter(all_blocked)
    assert Counter(results.outgoing) == Counter(all_blocked)
    # testing on a the list supported ports
    results = environment_generation.create_firewall_rules(potential_ports)
    assert Counter(results.incoming) == Counter(all_allowed)
    assert Counter(results.outgoing) == Counter(all_allowed)

    results = environment_generation.create_firewall_rules(half_ports)
    assert Counter(results.incoming) == Counter(half_allowed)
    assert Counter(results.outgoing) == Counter(half_allowed)
