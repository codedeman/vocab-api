"""The fixed topic taxonomy every vocabulary item is classified against.

Categories are predefined and closed: a classifier picks one of these, it never invents a
new main category. Subcategories are open strings, because the useful set differs per main
category and pinning them down now would be guesswork.
"""

from enum import Enum


class MainCategory(str, Enum):
    """The ten predefined main categories."""

    BUSINESS = "Business"
    WORKING_CAREER = "Working & Career"
    TECHNICAL_IT = "Technical & IT"
    DAILY_LIFE = "Daily Life & Communication"
    LEARNING = "Learning & Personal Development"
    FINANCE = "Finance"
    HEALTH = "Health & Wellbeing"
    SOCIETY_POLITICS = "Society & Politics"
    SCIENCE_ENVIRONMENT = "Science & Environment"
    OTHER = "Other"


#: Used when the context is too thin to commit to a real subcategory.
UNCATEGORIZED = "Uncategorized"

#: Valid CEFR levels, lowest to highest.
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
