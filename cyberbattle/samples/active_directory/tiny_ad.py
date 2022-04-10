from cyberbattle.simulation.model import FirewallConfiguration, FirewallRule, RulePermission
from cyberbattle.simulation import model as m
from typing import Dict


firewall_conf = FirewallConfiguration(
    [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW), FirewallRule("SHELL", RulePermission.ALLOW)],
    [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW), FirewallRule("SHELL", RulePermission.ALLOW)])


def default_vulnerabilities() -> m.VulnerabilityLibrary:
    lib = {}
    lib['FindDomainControllers'] = m.VulnerabilityInfo(
        description="Search for valid domain controllers in the current machines environment.",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.LeakedNodesId(nodes=["domain_controller_1"]),
        reward_string="Found domain controllers"
    )
    lib['EnumerateFileShares'] = m.VulnerabilityInfo(
        description="Find all SMB shares this machine knows about.",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.ExploitFailed(),
        reward_string="Found shares"
    )
    lib["ProbeAdmin"] = m.VulnerabilityInfo(
        description="Probe a workstation to see if you have admin creds on it",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.ProbeFailed(),
        reward_string="No admin creds."
    )
    lib['ScanForCreds'] = m.VulnerabilityInfo(
        description="Scan the local security managers for credentials. Need to be admin on the box.",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node="domain_controller_1", port="AD", credential="dc_1")]),
        precondition=m.Precondition("admin"),
        reward_string="DA credentials found"
    )
    return lib


def breach_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
    lib['AuthorizationSpoofAndCrack'] = m.VulnerabilityInfo(
        description="Spoof an authoritative source on the network to get a crackable hash, then try to crack it",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node="workstation_1", port="SHELL", credential="user_1")])
    )
    return lib


def admin_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
    lib["ProbeAdmin"] = m.VulnerabilityInfo(
        description="Probe a workstation to see if you have admin creds on it",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.ProbeSucceeded(discovered_properties=["admin"]),
        reward_string="Admin creds verified."
    )
    return lib


def dc_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
    lib['DumpNTDS'] = m.VulnerabilityInfo(
        description="Dump the NTDS file from AD",
        type=m.VulnerabilityType.LOCAL,
        outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node=f"workstation_{wid}", port='SHELL', credential=f'user_{uid}') for wid in range(0, 1) for uid in range(0, 20)]),
        precondition=m.Precondition("domain_controller"),
        reward_string="Dumped all user hashes. Get crackin'"
    )
    return lib


nodes = {
    "domain_controller_1": m.NodeInfo(services=[m.ListeningService(name="AD", allowedCredentials=["dc_1"])],
                                      properties=["domain_controller"],
                                      value=100,
                                      firewall=firewall_conf,
                                      vulnerabilities=dc_vulnerabilities(default_vulnerabilities())),
    "workstation_0": m.NodeInfo(services=[m.ListeningService(name="SHELL", allowedCredentials=[f"user_{uid}" for uid in range(0, 20)])],
                                value=0,
                                properties=["breach_node"],
                                vulnerabilities=breach_vulnerabilities(default_vulnerabilities()),
                                agent_installed=True,
                                firewall=firewall_conf,
                                reimagable=False),
    "workstation_1": m.NodeInfo(services=[m.ListeningService(name="SHELL", allowedCredentials=[f"user_{uid}" for uid in range(0, 20)])],
                                properties=["admin"],
                                value=1,
                                firewall=firewall_conf,
                                vulnerabilities=admin_vulnerabilities(default_vulnerabilities()))
}

global_vulnerability_library: Dict[m.VulnerabilityID, m.VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.Identifiers(
    properties=[
        'breach_node',
        'domain_controller',
        "admin"  # whether or not the users of this machine are admins
    ],
    ports=['SMB', 'AD', 'SHELL'],
    local_vulnerabilities=[
        'FindDomainControllers',
        'EnumerateFileShares',
        'AuthorizationSpoofAndCrack',
        'ScanForCreds',
        'DumpNTDS',
        'ProbeAdmin'
    ],
    remote_vulnerabilities=[
        'PasswordSpray'
    ]
)


def new_environment() -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=ENV_IDENTIFIERS
    )
