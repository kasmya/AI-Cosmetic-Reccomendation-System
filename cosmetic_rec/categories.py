"""Map free-text source categories onto one shared category vocabulary."""
import re

SKINCARE = ["Cleanser", "Toner", "Treatment", "Moisturizer", "Face Mask", "Sunscreen"]
MAKEUP = ["Foundation", "Concealer", "Powder", "Blush", "Lipstick", "Mascara", "Eyeshadow", "Eyeliner"]
ALL_CATEGORIES = SKINCARE + MAKEUP

# First match wins, so order matters (more specific rules first).
_RULES: list[tuple[str, str]] = [
    ("Sunscreen", r"sunscreen|sun protection|sun cream|suncare|sun care|spf"),
    ("Foundation", r"foundation|bb cream|cc cream|tinted moisturi"),
    ("Concealer", r"concealer|corrector"),
    ("Blush", r"blush|bronzer|highlighter"),
    ("Powder", r"face powder|compact powder|setting powder|loose powder"),
    ("Lipstick", r"lipstick|lip gloss|lip stain|lip colour|lip color|lip tint"),
    ("Mascara", r"mascara"),
    ("Eyeshadow", r"eye ?shadow|eye palette"),
    ("Eyeliner", r"eye ?liner|kohl|eye pencil"),
    ("Face Mask", r"face mask|sheet mask|clay mask|peel-off|masque"),
    ("Cleanser", r"cleans|face wash|micellar|makeup remover|make-up remover|facial wash"),
    ("Toner", r"toner|tonic|essence|facial mist"),
    ("Treatment", r"serum|anti-?aging|anti-?wrinkle|anti-?acne|acne|retinol|ampoule|treatment|exfoliat|peeling|"
                  r"eye cream|eye contour"),
    ("Moisturizer", r"moisturi|face cream|day cream|night cream|facial cream|hydrating cream|gel cream"),
]
_EXCLUDE = re.compile(r"\bhair|shampoo|conditioner|nail|body|hand |foot|shaving|deodorant|perfume|toothpaste|"
                      r"soap|bath|shower|baby|pet |depilat|after-?sun|lip balm|lip care|lip butter", re.I)


def map_category(text: str | None) -> str | None:
    """Shared category for a source category string, or None if it is not a face skincare/makeup product."""
    if not isinstance(text, str):
        return None
    t = text.lower()
    for category, pattern in _RULES:
        if re.search(pattern, t):
            # sunscreen is allowed even if the listing also says "body"
            if category != "Sunscreen" and _EXCLUDE.search(t):
                return None
            return category
    return None


# The Sephora skincare data files some makeup under skincare categories (a concealer listed as a "Treatment").
# Deliberately narrow: "complexion" and plain "corrector" are not matched because real skincare uses those words.
_MAKEUP_NAME = re.compile(
    r"concealer|under-?eye corrector|colou?r corrector|foundation|\bbb\b|\bcc\b|tinted|beauty balm|highlighter|"
    r"bronz|blush|lipstick|lip gloss|lip tint|mascara|eyeliner|\bbrow\b|eyeshadow|primer|full[- ]coverage|"
    r"skin tint", re.I)


def looks_like_makeup(name: str | None) -> bool:
    return bool(_MAKEUP_NAME.search(name or ""))
