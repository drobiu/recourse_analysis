from typing import Dict


class RecourseMethodData:
    def __init__(self, name: str, class_type: str, hyperparameters: Dict):
        self.name = name
        self.type = class_type
        self.hyperparameters = hyperparameters
        self.dataset = None
        self.factuals = None
        self.model = None
