# ner_utils.py
import re

def extract_entities(text):
    """
    Extracts numeric order IDs (4–10 digits) from user text.
    Returns a list of ("order_id", ID) tuples.
    """
    order_ids = re.findall(r"\b\d{4,10}\b", text)
    return [("order_id", oid) for oid in order_ids]
