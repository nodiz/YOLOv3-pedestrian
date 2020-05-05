import pandas as pd

# in case not using tensoboard


class Metrics(object):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def add_values(self, val):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def plot(self, **kwargs):
        raise NotImplementedError
