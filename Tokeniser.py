class SimpleTokeniser:
    def fromDataset(self, context: set[str]):
        tokens = {}
        counter = 0

        for sample in context:
            for word in sample.split():
                if word not in tokens:
                    tokens.update({word : counter + 1})
                    counter += 1
        return tokens