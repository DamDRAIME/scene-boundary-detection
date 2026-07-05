class SRTParsingError(ValueError):
    """Error linked to the parsing of a SRT file"""


class SubtitleExtractionError(Exception):
    """Error linked to the extraction of subtitles"""


class SubtitleParsingError(Exception):
    """Error linked to the parsing of a subtitles dataset (HDF5 file)"""
