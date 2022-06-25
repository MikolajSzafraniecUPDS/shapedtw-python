class SubsequenceShorterThanWindow(Exception):
    def __init__(self, subsequence_size: int, window_size: int):
        error_msg = "Subsequence length: {0}, window size: {1}".format(
            subsequence_size,
            window_size
        )
        super().__init__(error_msg)


class SubsequenceTooShort(Exception):
    def __init__(self, subsequence_size: int, min_required: int):
        error_msg = "Subsequence length: {0}, minimal required length: {1}".format(
            subsequence_size,
            min_required
        )
        super().__init__(error_msg)


class NotShapeDescriptor(Exception):
    def __init__(self, obj):
        error_msg = "Provided argument is object of class {0} instead of 'ShapeDescriptor'".format(
            obj.__class__.__name__
        )
        super().__init__(error_msg)


class WindowOfSizeOne(Exception):
    def __init__(self, descriptor_class):
        error_msg = "Window for {0} cannot be of size 1. Please specify another window width or subsequence length".format(
            descriptor_class
        )
        super().__init__(error_msg)


class WrongWeightsNumber(Exception):
    pass


class WrongPadType(Exception):
    pass


class NegativeSubsequenceWidth(Exception):
    pass