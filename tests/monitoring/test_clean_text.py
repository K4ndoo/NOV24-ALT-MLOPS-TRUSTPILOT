import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.text import clean_text

DAILY_INFERENCE = Path(__file__).resolve().parents[2] / "src" / "modelisation" / "daily_inference.py"

CASES = [
    "Très bonne expérience . RAS",
    "Voir https://www.trustpilot.com/review/xyz pour le détail",
    "Écrivez à service.client@exemple.fr svp",
    "Merci @LaRedoute et #livraison rapide",
    "   trop    d'espaces \n\n entre\tles mots   ",
    "MAJUSCULES et minuscules",
    "",
    None,
    np.nan,
]


def _production_clean_text():
    """Extrait clean_text de daily_inference.py sans importer le module (il charge torch)."""
    tree = ast.parse(DAILY_INFERENCE.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "clean_text")
    namespace = {"re": re, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAILY_INFERENCE), "exec"), namespace)
    return namespace["clean_text"]


@pytest.mark.parametrize("text", CASES)
def test_clean_text_identique_a_daily_inference(text):
    """La copie du monitoring doit rester identique à la fonction de production."""
    assert clean_text(text) == _production_clean_text()(text)
