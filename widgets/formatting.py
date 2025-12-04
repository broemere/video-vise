from fractions import Fraction


def frac_to_float(frac_str):
    """
    Turn a string fraction like "30000/1001" or "10/1"
    (or even "29.97") into a float.
    """
    if frac_str in (None, "", "N/A", "Unknown"):
        return 0.0
    try:
        return float(Fraction(frac_str))
    except ValueError:
        try:
            return float(frac_str)
        except ValueError:
            return 0.0

def format_decimal(value: float | None, max_decimals: int = 2) -> str:
    """
    Round to at most `max_decimals` places.
    If the result ends in .0, drop the decimal entirely.
    Returns "N/A" on None.
    """
    if value is None:
        return "N/A"
    # round and format as fixed-point
    s = f"{value:.{max_decimals}f}"
    # strip trailing zeros and then a trailing dot if present
    s = s.rstrip("0").rstrip(".")
    return s
