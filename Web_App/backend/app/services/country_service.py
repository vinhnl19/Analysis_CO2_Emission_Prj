from app.core.resources import df
def get_country_list():
    country_df = (
        df[['Country', 'ISO_Code', 'Continent']]
            .dropna()
            .drop_duplicates(subset=['Country'])
            .sort_values('Country')
        )

    return [
            {
                "country_name": row['Country'],
                "country_code": row['ISO_Code'],
                "continent": row['Continent']
            }
            for _, row in country_df.iterrows()
        ]