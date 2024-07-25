from enum import Enum


class AggregationMethods(Enum):
    AVERAGE = "average"
    MINIMUM = "minimum"
    SUMMATION = "summation"


class ValueBias(Enum):
    """
    Value bias: lower/upper bound of the confidence interval,
    or average value of the interval.
    """

    LOWER = "lower"
    AVERAGE = "average"
    UPPER = "upper"


class ValueFunctionOptions(Enum):
    LABELLED = "labelled"
    UNLABELLED = "unlabelled"
