class EvaluationPipeline:
    def __init__(self, evaluators: list):
        self.evaluators = evaluators

    def run_evalutaion(self, dataset: list):
        results = {}
        for entry in dataset:
            # Execute evaluations
            pass
        return results
