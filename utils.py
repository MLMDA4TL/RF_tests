def error_rate(er):
    """
    tranforms a sklearn score in error rate (in %)
    """
    return round((1 - er) * 100, 1)
