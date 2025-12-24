from app.core.resources import df
def get_range_year_from_data():
    return {
        "minYear": int(df["Year"].min()),
        "maxYear": int(df["Year"].max())
    }