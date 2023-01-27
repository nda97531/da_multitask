class DatasetNormaliser:
    def __init__(self, name: str, destination_folder: str, *args, **kwargs):
        """

        Args:
            name:
            destination_folder:
            *args:
            **kwargs:
        """
        self.name = name
        self.destination_folder = destination_folder
        self.args = args
        self.kwargs = kwargs

    def run(self):
        pass
