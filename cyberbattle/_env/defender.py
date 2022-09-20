# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Defines stock defender agents for the CyberBattle simulation.
"""
import random
import numpy
from abc import abstractmethod
from cyberbattle.simulation.model import Environment
from cyberbattle.simulation.actions import DefenderAgentActions
from ..simulation import model

import logging


class DefenderAgent:
    """Define the step function for a defender agent.
    Gets called after each step executed by the attacker agent."""

    @abstractmethod
    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        return None


class ScanAndReimageCompromisedMachines(DefenderAgent):
    """A defender agent that scans a subset of network nodes
     detects presence of an attacker on a given node with
    some fixed probability and if detected re-image the compromised node.

    probability -- probability that an attacker agent is detected when scanned given that the attacker agent is present
    scan_capacity -- maxium number of machine that a defender agent can scan in one simulation step
    scan_frequency -- frequencey of the scan in simulation steps
    """

    def __init__(self, probability: float, scan_capacity: int, scan_frequency: int):
        self.probability = probability
        self.scan_capacity = scan_capacity
        self.scan_frequency = scan_frequency

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        if t % self.scan_frequency == 0:
            # scan nodes at random
            scanned_nodes = random.choices(list(environment.network.nodes), k=self.scan_capacity)
            for node_id in scanned_nodes:
                node_info = environment.get_node(node_id)
                if node_info.status == model.MachineStatus.Running and \
                        node_info.agent_installed:
                    is_malware_detected = numpy.random.random() <= self.probability
                    if is_malware_detected:
                        if node_info.reimagable:
                            logging.info(f"Defender detected malware, reimaging node {node_id}")
                            actions.reimage_node(node_id)
                        else:
                            logging.info(f"Defender detected malware, but node cannot be reimaged {node_id}")


class ExternalRandomEvents(DefenderAgent):
    """A 'defender' that randomly alters network node configuration"""

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        self.patch_vulnerabilities_at_random(environment)
        self.stop_service_at_random(environment, actions)
        self.plant_vulnerabilities_at_random(environment)
        self.firewall_change_remove(environment)
        self.firewall_change_add(environment)

    def patch_vulnerabilities_at_random(self, environment: Environment, probability: float = 0.1) -> None:
        # Iterate through every node.
        for node_id, node_data in environment.nodes():
            # Have a boolean remove_vulnerability decide if we will remove one.
            remove_vulnerability = numpy.random.random() <= probability
            if remove_vulnerability and len(node_data.vulnerabilities) > 0:
                choice = random.choice(list(node_data.vulnerabilities))
                node_data.vulnerabilities.pop(choice)

    def stop_service_at_random(self, environment: Environment, actions: DefenderAgentActions, probability: float = 0.1) -> None:
        for node_id, node_data in environment.nodes():
            remove_service = numpy.random.random() <= probability
            if remove_service and len(node_data.services) > 0:
                service = random.choice(node_data.services)
                actions.stop_service(node_id, service.name)

    def plant_vulnerabilities_at_random(self, environment: Environment, probability: float = 0.1) -> None:
        for node_id, node_data in environment.nodes():
            add_vulnerability = numpy.random.random() <= probability
            # See all differences between current node vulnerabilities and global ones.
            new_vulnerabilities = numpy.setdiff1d(
                list(environment.vulnerability_library.keys()), list(node_data.vulnerabilities.keys()))
            # If we have decided that we will add a vulnerability and there are new vulnerabilities not already
            # on the node, then add them.
            if add_vulnerability and len(new_vulnerabilities) > 0:
                new_vulnerability = random.choice(new_vulnerabilities)
                node_data.vulnerabilities[new_vulnerability] = \
                    environment.vulnerability_library[new_vulnerability]

    """
    TODO: Not sure how to access global (environment) services.
    def serviceChangeAdd(self, probability: float) -> None:
        # Iterate through every node.
        for node_id, node_data in self.__environment.nodes():
            # Have a boolean addService decide if we will add one.
            addService = numpy.random.random() <= probability
            # List all new services we can add.
            newServices = numpy.setdiff1d(self.__environment.services, node_data.services)
            # If we have decided to add a service and there are new services to add, go ahead and add them.
            if addService and len(newServices) > 0:
                newService = random.choice(newServices)
                node_data.services.append(newService)
        return None
    """

    def firewall_change_remove(self, environment: Environment, probability: float = 0.1) -> None:
        # Iterate through every node.
        for node_id, node_data in environment.nodes():
            # Have a boolean remove_rule decide if we will remove one.
            remove_rule = numpy.random.random() <= probability
            # The following logic sees if there are both incoming and outgoing rules.
            # If there are, we remove one randomly.
            if remove_rule and len(node_data.firewall.outgoing) > 0 and len(node_data.firewall.incoming) > 0:
                incoming = numpy.random.random() <= 0.5
                if incoming:
                    rule_to_remove = random.choice(node_data.firewall.incoming)
                    node_data.firewall.incoming.remove(rule_to_remove)
                else:
                    rule_to_remove = random.choice(node_data.firewall.outgoing)
                    node_data.firewall.outgoing.remove(rule_to_remove)
            # If there are only outgoing rules, we remove one random outgoing rule.
            elif remove_rule and len(node_data.firewall.outgoing) > 0:
                rule_to_remove = random.choice(node_data.firewall.outgoing)
                node_data.firewall.outgoing.remove(rule_to_remove)
            # If there are only incoming rules, we remove one random incoming rule.
            elif remove_rule and len(node_data.firewall.incoming) > 0:
                rule_to_remove = random.choice(node_data.firewall.incoming)
                node_data.firewall.incoming.remove(rule_to_remove)

    def firewall_change_add(self, environment: Environment, probability: float = 0.1) -> None:
        # Iterate through every node.
        for node_id, node_data in environment.nodes():
            # Have a boolean rule_to_add decide if we will add one.
            add_rule = numpy.random.random() <= probability
            if add_rule:
                # 0 For allow, 1 for block.
                rule_to_add = model.FirewallRule(port=random.choice(model.SAMPLE_IDENTIFIERS.ports),
                                                 permission=model.RulePermission.ALLOW)
                # Randomly decide if we will add an incoming or outgoing rule.
                incoming = numpy.random.random() <= 0.5
                if incoming and rule_to_add not in node_data.firewall.incoming:
                    node_data.firewall.incoming.append(rule_to_add)
                elif not incoming and rule_to_add not in node_data.firewall.incoming:
                    node_data.firewall.outgoing.append(rule_to_add)
