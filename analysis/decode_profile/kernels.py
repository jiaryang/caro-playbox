"""Kernel name classification driven by an editable CSV rule file."""

from __future__ import annotations

import csv
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # analysis/
GLM52_RULES = os.path.join(HERE, "rules", "glm52.csv")
BASE_RULES = os.path.join(HERE, "rules", "kernel_categories.csv")

OTHER = "other"


class KernelClassifier:
    """First matching rule wins, so row order in the CSV is the priority order."""

    def __init__(self, rules_path: str = GLM52_RULES):
        self.rules_path = rules_path
        self.categories = []
        self._compiled = []
        with open(rules_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                category = (row.get("category") or "").strip()
                pattern = (row.get("pattern") or "").strip()
                if not category or not pattern:
                    continue
                self.categories.append(category)
                self._compiled.append((category, re.compile(pattern)))
        self.categories.append(OTHER)
        self._cache = {}

    def classify(self, name: str) -> str:
        hit = self._cache.get(name)
        if hit is None:
            low = name.lower()
            hit = OTHER
            for category, pattern in self._compiled:
                if pattern.search(low):
                    hit = category
                    break
            self._cache[name] = hit
        return hit

    def order_index(self, category: str) -> int:
        try:
            return self.categories.index(category)
        except ValueError:
            return len(self.categories)
