from sbd.screenplay.typings import Label


def sanitize_labels(labels_raw: str, labels_to_ignore: frozenset[Label]) -> frozenset[Label]:
    labels = {Label[label.strip()] for label in labels_raw.split(",")}
    reduced = labels - labels_to_ignore
    return frozenset(reduced) if reduced else frozenset(labels)
