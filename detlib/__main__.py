import fire
# TODO: Optional but could be useful to use fire for CLI scripts (train/test etc)

class CLI:
    """
    Detlib package

    ---------------
    examples:
        >>> python -m detlib --help
    """

    def evaluate(self, *args):
        return ...

    def train(self, *args):
        return ...


if __name__ == "__main__":
    fire.Fire(CLI)
