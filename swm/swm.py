from data import Consensus, Action

class SimpleSWM:
    def __init__(self, model_name: str) -> None:
        self.model = model_name

    def predict(self, consensus: Consensus, src_ts: int, tgt_ts: int) -> int:
        prompt = "Please predict the possibility of the following event happening: " + consensus.question + " Answer: " + consensus.answer
        
