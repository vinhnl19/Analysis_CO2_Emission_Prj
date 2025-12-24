def format_number(value, decimals=None):
    if value is None:
        return "--"
    if decimals is None:
        return f"{value:,}"
    return f"{value:,.{decimals}f}"
