"""Financial Institutions"""
# pylint: disable=C0301
from typing import cast, Iterator
from cyberbattle.simulation import model as m

default_allow_rules = []

# Subnet configuration
subnets = m.Subnets(['Internet(outer)', 'Internet(inner)', 'Intranet', 'DMZ', 'Core'])

allow_ports = ['HTTP', 'HTTPS', 'IMAP', 'SMTP']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Internet(outer)', 'DMZ', m.SubnetConnectivity(conn=True, firewall=rules))
subnets.set_conn('DMZ', 'Internet(outer)', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTP', 'HTTPS']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Internet(outer)', 'Internet(inner)', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTP', 'HTTPS']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Internet(inner)', 'Internet(outer)', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['USB']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Internet(inner)', 'Intranet', m.SubnetConnectivity(conn=True, firewall=rules))
subnets.set_conn('Intranet', 'Internet(inner)', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTPS', 'SQL']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('DMZ', 'Core', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTPS']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Core', 'DMZ', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTPS', 'SQL', 'SSH', 'FTP']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Intranet', 'Core', m.SubnetConnectivity(conn=True, firewall=rules))

allow_ports = ['HTTPS']
rules = [ m.FirewallRule(port, m.RulePermission.ALLOW) for port in allow_ports ]
subnets.set_conn('Core', 'Intranet', m.SubnetConnectivity(conn=True, firewall=rules))

# Network nodes in a financial institution

nodes = {
    'Client': m.NodeInfo(
        services = [],
        subnet = 'Internet(outer)',
        vulnerabilities = {
            'T1217': m.VulnerabilityInfo(
                description='T1217: Browser Information Discovery',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['WebServer']),
                reward_string='Web browser information contains history where website URLs of interest are',
                cost=1.0,
            )
        }
    ),
    'WebServer': m.NodeInfo(
        services = [
            m.ListeningService('HTTP'),
            m.ListeningService('HTTPS'),
            m.ListeningService('SSH', allowedCredentials=['WebManager'])
        ],
        subnet = 'DMZ',
        properties = ['OS_Linux', 'SW_WebServer'],
        vulnerabilities = {
            'T1594': m.VulnerabilityInfo(
                description='T1594: Search Victim-Owned Websites',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(['MailServer']),
                port=['HTTP', 'HTTPS'],
                reward_string='Webpage contains information about the institution\'s mail server',
                cost=1.0
            )
        }
    ),
    'MailServer': m.NodeInfo(
        services = [
            m.ListeningService('SMTP'),
            m.ListeningService('IMAP')
        ],
        subnet = 'DMZ',
        properties = ['OS_Linux', 'SW_MailServer'],
        vulnerabilities = {
            'T1210': m.VulnerabilityInfo(
                description='T1210: Exploitation of Remote Services',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                port=['SMTP'],
                reward_string='',
                cost=1.0
            ),
            'T1040': m.VulnerabilityInfo(
                description='T1040: Network Sniffing',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC', 'StorageServer', 'WebServer']),
                reward_string='Network traffic contains nodes\' information, which use the mail server',
                cost=1.0
            ),
        }
    ),
    'InternetPC': m.NodeInfo(
        services = [m.ListeningService('USB')],
        subnet = 'Internet(inner)',
        properties = ['OS_Windows'],
        vulnerabilities = {
            'T1566.001': m.VulnerabilityInfo(
                description='T1566.001: Spearphishing Attachment',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                reward_string='',
                cost=1.0
            ),
            'T1566.002': m.VulnerabilityInfo(
                description='T1566.002: Spearphishing Link',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                reward_string='',
                cost=1.0
            ),
            'T1566.003': m.VulnerabilityInfo(
                description='T1566.003: Spearphishing via Service',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                reward_string='',
                cost=1.0
            ),
            'T1566.004': m.VulnerabilityInfo(
                description='T1566.004: Spearphishing Voice',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials([
                    m.CachedCredential(node='WebServer', port='SSH', credential='Cred_WebManager')
                ]),
                reward_string='',
                cost=1.0
            ),
            'T1091': m.VulnerabilityInfo(
                description='T1091: Replication Through Removable Media',
                type=m.VulnerabilityType.REMOTE,
                port=['USB'],
                outcome=m.LateralMove(),
                reward_string = '',
                cost = 1.0
            ),
            'T1652': m.VulnerabilityInfo(
                description='T1652: Device Driver Discovery',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['IntraWorker', 'IntraManager', 'IntraSysEng']),
                reward_string = '',
                cost = 1.0
            ),
            'T1654': m.VulnerabilityInfo(
                description='T1654: Log Enumeration',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['IntraWorker', 'IntraManager', 'IntraSysEng']),
                reward_string = '',
                cost = 1.0
            )
        }
    ),
    'IntraWorker': m.NodeInfo(
        services = [m.ListeningService('USB')],
        subnet = 'Intranet',
        properties = ['OS_Windows'],
        vulnerabilities={
            'T1091': m.VulnerabilityInfo(
                description='T1091: Replication Through Removable Media',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                port=['USB'],
                reward_string = '',
                cost = 1.0
            ),
            'T1652': m.VulnerabilityInfo(
                description='T1652: Device Driver Discovery',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            ),
            'T1654': m.VulnerabilityInfo(
                description='T1654: Log Enumeration',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            ),
            'T1595.001': m.VulnerabilityInfo(
                description='T1595.001: Scanning IP Blocks',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId([''])
            )
        }
    ),
    'IntraManager': m.NodeInfo(
        services = [m.ListeningService('USB')],
        subnet = 'Intranet',
        properties = ['OS_Windows'],
        vulnerabilities={
            'T1091': m.VulnerabilityInfo(
                description='T1091: Replication Through Removable Media',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                port=['USB'],
                reward_string = '',
                cost = 1.0
            ),
            'T1652': m.VulnerabilityInfo(
                description='T1652: Device Driver Discovery',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            ),
            'T1654': m.VulnerabilityInfo(
                description='T1654: Log Enumeration',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            ),
            'T1595.001': m.VulnerabilityInfo(
                description='T1595.001: Scanning IP Blocks',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId([''])
            )
        }
    ),
    'IntraSysEng': m.NodeInfo(
        services = [m.ListeningService('USB')],
        subnet = 'Intranet',
        properties = ['OS_Windows'],
        vulnerabilities={
            'T1091': m.VulnerabilityInfo(
                description='T1091: Replication Through Removable Media',
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LateralMove(),
                port=['USB'],
                reward_string = '',
                cost = 1.0
            ),
            'T1652': m.VulnerabilityInfo(
                description='T1652: Device Driver Discovery',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            ),
            'T1654': m.VulnerabilityInfo(
                description='T1654: Log Enumeration',
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(['InternetPC']),
                reward_string = '',
                cost = 1.0
            )
        }
    ),
    'CoreServer': m.NodeInfo(
        services = [
            m.ListeningService('HTTPS'),
            m.ListeningService('SSH', allowedCredentials=['Cred_SysEng'])
        ],
        subnet = 'Core',
        properties = ['OS_Linux', 'APIServer', 'BusinessServer']
    ),
    'StorageServer': m.NodeInfo(
        services = [
            m.ListeningService('SQL', allowedCredentials=['Cred_SysEng', 'Cred_MailServer', 'Cred_WebServer', 'Cred_CoreServer']),
            m.ListeningService('FTP', allowedCredentials=['Cred_SysEng']),
            m.ListeningService('SSH', allowedCredentials=['Cred_SysEng'])
        ],
        subnet = 'Core',
        properties = ['OS_Linux', 'FTPServer', 'SQLServer']
    )
}

global_vulnerability_library: dict[m.VulnerabilityID, m.VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.infer_constants_from_nodes(cast(Iterator[tuple[m.NodeID, m.NodeInfo]], list(nodes.items())), global_vulnerability_library)

def new_environment() -> m.Environment:
    """Creates new financial institution environment"""
    return m.Environment(network=m.create_network(nodes), vulnerability_library=global_vulnerability_library, identifiers=ENV_IDENTIFIERS, subnets=subnets)
