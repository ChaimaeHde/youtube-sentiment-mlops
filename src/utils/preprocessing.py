import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+|\b[ur]/\w+")      # @user, u/user, r/subreddit
HTML_RE = re.compile(r"<.*?>")
SPECIAL_RE = re.compile(r"[^\w\s!?']")            # garde lettres, chiffres, ! ? '
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Nettoie un commentaire (identique à l'entraînement et à l'inférence)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = text.replace("_", " ")
    return WS_RE.sub(" ", text).strip()