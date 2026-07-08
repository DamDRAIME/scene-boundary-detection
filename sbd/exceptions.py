class MHTMLParsingError(ValueError):
    """Error linked to the parsing of a MHTML file"""


class VideoParsingError(ValueError):
    """Error linked to the parsing of a video"""


class SpriteExtractionError(ValueError):
    """Error linked to the extraction of frames/sprites"""


class SRTParsingError(ValueError):
    """Error linked to the parsing of a SRT file"""


class SubtitleExtractionError(Exception):
    """Error linked to the extraction of subtitles"""
