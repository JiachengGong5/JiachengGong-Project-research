"""CICIoT2023 defaults for the configurable hierarchical label system."""

from __future__ import annotations

from .schema import LabelSchema


CATEGORY_TO_FINE_LABELS: dict[str, tuple[str, ...]] = {
    "Benign": ("Benign",),
    "DDoS": (
        "DDoS-ACK fragmentation",
        "DDoS-UDP flood",
        "DDoS-SlowLoris",
        "DDoS-ICMP flood",
        "DDoS-RSTFIN flood",
        "DDoS-PSHACK flood",
        "DDoS-HTTP flood",
        "DDoS-UDP fragmentation",
        "DDoS-TCP flood",
        "DDoS-SYN flood",
        "DDoS-SynonymousIP flood",
        "DDoS-ICMP fragmentation",
    ),
    "DoS": (
        "DoS-TCP flood",
        "DoS-HTTP flood",
        "DoS-SYN flood",
        "DoS-UDP flood",
    ),
    "Recon": (
        "Recon-Ping sweep",
        "Recon-OS scan",
        "Recon-Vulnerability scan",
        "Recon-Port scan",
        "Recon-Host discovery",
    ),
    "Web-based": (
        "Web-Sql injection",
        "Web-Command injection",
        "Web-Backdoor malware",
        "Web-Uploading attack",
        "Web-XSS",
        "Web-Browser hijacking",
    ),
    "Brute Force": ("BruteForce-Dictionary brute force",),
    "Spoofing": (
        "Spoofing-Arp spoofing",
        "Spoofing-DNS spoofing",
    ),
    "Mirai": (
        "Mirai-GREIP flood",
        "Mirai-Greeth flood",
        "Mirai-UDPPlain",
    ),
}

CATEGORY_NAMES: tuple[str, ...] = tuple(CATEGORY_TO_FINE_LABELS)
FINE_LABEL_TO_CATEGORY: dict[str, str] = {
    fine_label: category
    for category, fine_labels in CATEGORY_TO_FINE_LABELS.items()
    for fine_label in fine_labels
}
FINE_LABEL_NAMES: tuple[str, ...] = tuple(FINE_LABEL_TO_CATEGORY)
CATEGORY_TO_NUM_FINE: dict[str, int] = {
    category: len(fine_labels)
    for category, fine_labels in CATEGORY_TO_FINE_LABELS.items()
}


_CICIOT2023_ALIASES: dict[str, str] = {
    "Benign_Final": "Benign",
    "BenignTraffic": "Benign",
    "Backdoor_Malware": "Web-Backdoor malware",
    "BrowserHijacking": "Web-Browser hijacking",
    "CommandInjection": "Web-Command injection",
    "SqlInjection": "Web-Sql injection",
    "Uploading_Attack": "Web-Uploading attack",
    "XSS": "Web-XSS",
    "DictionaryBruteForce": "BruteForce-Dictionary brute force",
    "DNS_Spoofing": "Spoofing-DNS spoofing",
    "MITM-ArpSpoofing": "Spoofing-Arp spoofing",
    "Recon-HostDiscovery": "Recon-Host discovery",
    "Recon-OSScan": "Recon-OS scan",
    "Recon-PingSweep": "Recon-Ping sweep",
    "Recon-PortScan": "Recon-Port scan",
    "VulnerabilityScan": "Recon-Vulnerability scan",
    "Mirai-greeth_flood": "Mirai-Greeth flood",
    "Mirai-greip_flood": "Mirai-GREIP flood",
    "Mirai-udpplain": "Mirai-UDPPlain",
}

CICIOT2023_SCHEMA = LabelSchema(
    name="CICIoT2023",
    category_to_fine_labels=CATEGORY_TO_FINE_LABELS,
    aliases=_CICIOT2023_ALIASES,
)


def fine_index_within_category(fine_label: str) -> int:
    """Return the CICIoT2023 category-local index for ``fine_label``."""

    return CICIOT2023_SCHEMA.fine_index_within_category(fine_label)


def fine_label_from_category_index(category: str, fine_index: int) -> str:
    """Return the CICIoT2023 fine label for a category-local index."""

    return CICIOT2023_SCHEMA.fine_label_from_category_index(category, fine_index)


def category_index(category: str) -> int:
    """Return the CICIoT2023 coarse-label index."""

    return CICIOT2023_SCHEMA.category_index(category)


def canonical_fine_label(label: str) -> str:
    """Normalize a dataset folder/file label to the canonical fine label."""

    return CICIOT2023_SCHEMA.canonical_fine_label(label)


def coarse_label_for_fine(fine_label: str) -> str:
    """Return the coarse category for a raw or canonical fine label."""

    return CICIOT2023_SCHEMA.coarse_label_for_fine(fine_label)
