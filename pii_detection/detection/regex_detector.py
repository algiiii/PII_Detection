from __future__ import annotations

import re

from pii_detection.detection.protocol import BaseDetector
from pii_detection.detection.types import TextSpan, PIICandidate, DetectorKind
from pii_detection.detection.config import RegexRuleModel

class RegexDetector(BaseDetector):
    