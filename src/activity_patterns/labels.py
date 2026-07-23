"""CICIoT2023 hierarchy definitions for grouped and fine-grained labels."""

from __future__ import annotations

import re


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


def fine_index_within_category(fine_label: str) -> int:
    """Return the class index used by the fine head for ``fine_label``."""

    category = FINE_LABEL_TO_CATEGORY[fine_label]
    return CATEGORY_TO_FINE_LABELS[category].index(fine_label)


def fine_label_from_category_index(category: str, fine_index: int) -> str:
    """Return the fine label represented by a category-local class index."""

    return CATEGORY_TO_FINE_LABELS[category][fine_index]


def category_index(category: str) -> int:
    """Return the coarse-label index used by the hierarchical model."""

    return CATEGORY_NAMES.index(category)


def _label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


_FINE_LABEL_ALIASES: dict[str, str] = {
    _label_key(fine_label): fine_label for fine_label in FINE_LABEL_NAMES
}
_FINE_LABEL_ALIASES.update(
    {
        _label_key("Benign_Final"): "Benign",
        _label_key("BenignTraffic"): "Benign",
        _label_key("Backdoor_Malware"): "Web-Backdoor malware",
        _label_key("BrowserHijacking"): "Web-Browser hijacking",
        _label_key("CommandInjection"): "Web-Command injection",
        _label_key("SqlInjection"): "Web-Sql injection",
        _label_key("Uploading_Attack"): "Web-Uploading attack",
        _label_key("XSS"): "Web-XSS",
        _label_key("DictionaryBruteForce"): "BruteForce-Dictionary brute force",
        _label_key("DNS_Spoofing"): "Spoofing-DNS spoofing",
        _label_key("MITM-ArpSpoofing"): "Spoofing-Arp spoofing",
        _label_key("Recon-HostDiscovery"): "Recon-Host discovery",
        _label_key("Recon-OSScan"): "Recon-OS scan",
        _label_key("Recon-PingSweep"): "Recon-Ping sweep",
        _label_key("Recon-PortScan"): "Recon-Port scan",
        _label_key("VulnerabilityScan"): "Recon-Vulnerability scan",
        _label_key("Mirai-greeth_flood"): "Mirai-Greeth flood",
        _label_key("Mirai-greip_flood"): "Mirai-GREIP flood",
        _label_key("Mirai-udpplain"): "Mirai-UDPPlain",
    }
)


def canonical_fine_label(label: str) -> str:
    """Normalize a dataset folder/file label to the canonical fine label."""

    key = _label_key(label)
    if key not in _FINE_LABEL_ALIASES:
        raise KeyError(f"Unknown CICIoT2023 fine label: {label!r}")
    return _FINE_LABEL_ALIASES[key]


def coarse_label_for_fine(fine_label: str) -> str:
    """Return the coarse category for a raw or canonical fine label."""

    return FINE_LABEL_TO_CATEGORY[canonical_fine_label(fine_label)]
