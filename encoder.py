from sklearn.preprocessing import LabelEncoder


class Encoder:
    """Simple class to go between actual and encoded IDs."""

    def __init__(self, original):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(original)
        self.original2encoded = {original: encoded for original, encoded in zip(original, encoded)}
        self.encoded2original = {encoded: original for original, encoded in zip(original, encoded)}

    def original(self, encoded):
        return self.encoded2original[encoded]

    def encoded(self, original):
        return self.original2encoded[original]
