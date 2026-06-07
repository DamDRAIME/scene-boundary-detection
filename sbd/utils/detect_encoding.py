from pathlib import Path

import chardet


def detect_encoding(filepath: Path, default: str = "utf-8") -> str:
    detector = chardet.UniversalDetector()
    with filepath.open("rb") as fh:
        chunk = fh.read(4096)
        while chunk and not detector.done:
            detector.feed(chunk)
            chunk = fh.read(4096)

    detector.close()
    charset = detector.result["encoding"]
    return charset if charset else default
