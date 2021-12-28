
class BaseTransform(object):

    def __init__(self, p:float) -> None:
        super().__init__()
        self._check_p_if_valid(p)
        self.p = p
        self.t = None

    def _check_p_if_valid(self, p):
        assert p < 1 and p > 0, "p should in range (0,1)"

    def __call__(self, data_sample):
        return NotImplementedError

    @property
    def transform(self):
        return self.t