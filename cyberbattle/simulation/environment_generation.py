# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
    environment_generation.py this function generates a semi random environment for
    the loonshot simulation v0.
"""
from typing import List, Dict, Set
import random
import re
import networkx as nx
from . import model

# These two lists are lists of potential vulnerabilities. They are split into linux vulnerabilities
# and Windows vulnerabilities so i can

ADMINTAG = model.AdminEscalation().tag
SYSTEMTAG = model.SystemEscalation().tag

potential_windows_vulns = {
    "UACME43":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #43",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&(Win10|Win7)&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "UACME45":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #45",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "UACME52":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #52",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&(Win10|Win7)&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "UACME55":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #55",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&(Win10|Win7)&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "UACME61":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #61",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "MimikatzLogonpasswords":
    model.VulnerabilityInfo(
        description="Mimikatz sekurlsa::logonpasswords.",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/gentilkiwi/mimikatz",
        precondition=model.Precondition(f"Windows&({ADMINTAG}|{SYSTEMTAG})"),
        outcome=model.LeakedCredentials([]),
        rates=model.Rates(0, 1.0, 1.0)),
    "MimikatzKerberosExport":
    model.VulnerabilityInfo(
        description="Mimikatz Kerberos::list /export."
                    "Exports .kirbi files to be used with pass the ticket",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/gentilkiwi/mimikatz",
        precondition=model.Precondition(f"Windows&DomainJoined&({ADMINTAG}|{SYSTEMTAG})"),
        outcome=model.LeakedCredentials([]),
        rates=model.Rates(0, 1.0, 1.0)),
    "PassTheTicket":
    model.VulnerabilityInfo(
        description="Mimikatz Kerberos::ptt /export."
                    "Exports .kirbi files to be used with pass the ticket",
        type=model.VulnerabilityType.REMOTE,
        URL="https://github.com/gentilkiwi/mimikatz",
        precondition=model.Precondition(f"Windows&DomainJoined&KerberosTicketsDumped"
                                        f"&({ADMINTAG}|{SYSTEMTAG})"),
        outcome=model.LeakedCredentials([]),
        rates=model.Rates(0, 1.0, 1.0)),
    "RDPBF":
    model.VulnerabilityInfo(
        description="RDP Brute Force",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1110/",
        precondition=model.Precondition("Windows&PortRDPOpen"),
        outcome=model.LateralMove(),
        rates=model.Rates(0, 0.2, 1.0)),

    "SMBBF":
    model.VulnerabilityInfo(
        description="SSH Brute Force",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1110/",
        precondition=model.Precondition("(Windows|Linux)&PortSMBOpen"),
        outcome=model.LateralMove(),
        rates=model.Rates(0, 0.2, 1.0))
}

potential_linux_vulns = {
    "SudoCaching":
    model.VulnerabilityInfo(
        description="Escalating privileges from poorly configured sudo on linux/unix machines",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1206/",
        precondition=model.Precondition(f"Linux&(~{ADMINTAG})"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 1.0, 1.0)),
    "SSHBF":
    model.VulnerabilityInfo(
        description="SSH Brute Force",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1110/",
        precondition=model.Precondition("Linux&PortSSHOpen"),
        outcome=model.LateralMove(),
        rates=model.Rates(0, 0.2, 1.0)),
    "SMBBF":
    model.VulnerabilityInfo(
        description="SSH Brute Force",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1110/",
        precondition=model.Precondition("(Windows|Linux)&PortSMBOpen"),
        outcome=model.LateralMove(),
        rates=model.Rates(0, 0.2, 1.0))
}

# These are potential endpoints that can be open in a game. Note to add any more endpoints simply
# add the protocol name to this list.
# further note that ports are stored in a tuple. This is because some protoocls
# (like SMB) have multiple official ports.
potential_ports: List[model.PortName] = ["RDP", "SSH", "HTTP", "HTTPs",
                                         "SMB", "SQL", "FTP", "WMI"]

# These two lists are potential node states. They are split into linux states and windows
#  states so that we can generate real graphs that aren't just totally random.
potential_linux_node_states: List[model.PropertyName] = ["Linux", ADMINTAG,
                                                         "PortRDPOpen",
                                                         "PortHTTPOpen", "PortHTTPsOpen",
                                                         "PortSSHOpen", "PortSMBOpen",
                                                         "PortFTPOpen", "DomainJoined"]
potential_windows_node_states: List[model.PropertyName] = ["Windows", "Win10", "PortRDPOpen",
                                                           "PortHTTPOpen", "PortHTTPsOpen",
                                                           "PortSSHOpen", "PortSMBOpen",
                                                           "PortFTPOpen", "BITSEnabled",
                                                           "Win7", "DomainJoined"]

ENV_IDENTIFIERS = model.Identifiers(
    ports=potential_ports,
    properties=potential_linux_node_states + potential_windows_node_states,
    local_vulnerabilities=list(potential_windows_vulns.keys()),
    remote_vulnerabilities=list(potential_windows_vulns.keys())
)


def create_random_environment(name: str, size: int) -> model.Environment:
    """
        This is the create random environment function. It takes a string for the name
        of the environment and an int for the size. It returns a randomly genernated
        environment.

        Note this does not currently support generating credentials.
    """
    if not name:
        raise ValueError("Please supply a non empty string for the name")

    if size < 1:
        raise ValueError("Please supply a positive non zero positive"
                         "integer for the size of the environment")
    graph = nx.DiGraph()
    nodes: Dict[str, model.NodeInfo] = {}

    # append the linux and windows vulnerability dictionaries
    local_vuln_lib: Dict[model.VulnerabilityID, model.VulnerabilityInfo] = \
        {**potential_windows_vulns, **potential_linux_vulns}

    os_types: List[str] = ["Linux", "Windows"]
    for i in range(size):
        rand_os: str = os_types[random.randint(0, 1)]
        nodes[str(i)] = create_random_node(rand_os, potential_ports)

    nodes['0'].agent_installed = True

    graph.add_nodes_from([(k, {'data': v}) for (k, v) in list(nodes.items())])

    return model.Environment(network=graph, vulnerability_library=local_vuln_lib, identifiers=ENV_IDENTIFIERS)


def create_random_node(os_type: str, end_points: List[model.PortName]) \
        -> model.NodeInfo:
    """
        This is the create random node function.
        Currently it takes a string for the OS type and returns a NodeInfo object
        Options for OS type are currently Linux or Windows,
        Options for the role are Server or Workstation
    """

    if not end_points:
        raise ValueError("No endpoints supplied")

    if os_type not in ("Windows", "Linux"):
        raise ValueError("Unsupported OS Type please enter Linux or Windows")

    # get the vulnerability dictionary for the important OS
    vulnerabilities: model.VulnerabilityLibrary = dict([])
    if os_type == "Linux":
        vulnerabilities = \
            select_random_vulnerabilities(os_type, random.randint(1, len(potential_linux_vulns)))
    else:
        vulnerabilities = \
            select_random_vulnerabilities(os_type, random.randint(1, len(potential_windows_vulns)))

    firewall: model.FirewallConfiguration = create_firewall_rules(end_points)
    properties: List[model.PropertyName] = \
        get_properties_from_vulnerabilities(os_type, vulnerabilities)

    return model.NodeInfo(services=[model.ListeningService(name=p) for p in end_points],
                          vulnerabilities=vulnerabilities,
                          value=int(random.random()),
                          properties=properties,
                          firewall=firewall,
                          agent_installed=False)


def select_random_vulnerabilities(os_type: str, num_vulns: int) \
        -> Dict[str, model.VulnerabilityInfo]:
    """
        It takes an a string for the OS type,  and an int for the number of
        vulnerabilities to select.

        It selects num_vulns vulnerabilities from the global list of vulnerabilities for that
        specific operating system.  It returns a dictionary of VulnerabilityInfo objects to
        the caller.
    """

    if num_vulns < 1:
        raise ValueError("Expected a positive value for num_vulns in select_random_vulnerabilities")

    ret_val: Dict[str, model.VulnerabilityInfo] = {}
    keys: List[str]
    if os_type == "Linux":
        keys = random.sample(list(potential_linux_vulns.keys()), num_vulns)
        ret_val = {k: potential_linux_vulns[k] for k in keys}
    elif os_type == "Windows":
        keys = random.sample(list(potential_windows_vulns.keys()), num_vulns)
        ret_val = {k: potential_windows_vulns[k] for k in keys}
    else:
        raise ValueError("Invalid Operating System supplied to select_random_vulnerabilities")
    return ret_val


def get_properties_from_vulnerabilities(os_type: str,
                                        vulns: Dict[model.NodeID, model.VulnerabilityInfo]) \
        -> List[model.PropertyName]:
    """
        get_properties_from_vulnerabilities function.
        This function takes a string for os_type and returns a list of PropertyName objects
    """
    ret_val: Set[model.PropertyName] = set()
    properties: List[model.PropertyName] = []

    if os_type == "Linux":
        properties = potential_linux_node_states
    elif os_type == "Windows":
        properties = potential_windows_node_states

    for prop in properties:

        for vuln_id, vuln in vulns.items():
            if re.search(prop, str(vuln.precondition.expression)):
                ret_val.add(prop)

    return list(ret_val)


def create_firewall_rules(end_points: List[model.PortName]) -> model.FirewallConfiguration:
    """
        This function takes a List of endpoints and returns a FirewallConfiguration

        It iterates through the list of potential ports and if they're in the list passed
        to the function it adds a firewall rule allowing that port.
        Otherwise it adds a rule blocking that port.
    """

    ret_val: model.FirewallConfiguration = model.FirewallConfiguration()
    ret_val.incoming.clear()
    ret_val.outgoing.clear()
    for protocol in potential_ports:
        if protocol in end_points:
            ret_val.incoming.append(model.FirewallRule(protocol, model.RulePermission.ALLOW))
            ret_val.outgoing.append(model.FirewallRule(protocol, model.RulePermission.ALLOW))
        else:
            ret_val.incoming.append(model.FirewallRule(protocol, model.RulePermission.BLOCK))
            ret_val.outgoing.append(model.FirewallRule(protocol, model.RulePermission.BLOCK))

    return ret_val
