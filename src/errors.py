class SingletonError(Exception):
    """Custom exception for singleton class."""
    def __init__(self, instance):  
        self.msg = f'An instance of class {instance} already exists. Please delete/cleanup the existing instance before creating a new one!'
        super().__init__(self.msg)


class OutOfBoundsError(Exception):
    """Custom exception for out of bounds values."""
    def __init__(self, values: tuple, bounds: tuple):            
        self.msg = f'Values out of bounds!\n\t{values=}\n\t{bounds=}'
        super().__init__(self.msg)