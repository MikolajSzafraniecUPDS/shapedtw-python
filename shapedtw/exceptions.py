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