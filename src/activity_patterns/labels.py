"""CICIoT2023 hierarchy definitions for grouped and fine-grained labels."""

from __future__ import annotations


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

