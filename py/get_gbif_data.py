from pygbif import occurrences as gbif_occurrences, species as gbif_species
import pandas as pd
import time

def get_gbif_data(scientific_names, area_bbox=None, limit_per_species=1000):
    """
    scientific_names: list of taxon scientific names
    area_bbox: [min_lon, min_lat, max_lon, max_lat]
    Returns combined pandas DataFrame with columns lat, lon, time
    """
    frames = []
    for name in scientific_names:
        params = {
            'scientificName': name,
            'limit': limit_per_species,
            'hasCoordinate': True,
        }
        if area_bbox:
            params['decimalLatitude'] = f"{area_bbox[1]},{area_bbox[3]}"
            params['decimalLongitude'] = f"{area_bbox[0]},{area_bbox[2]}"
        results = gbif_occurrences.search(**params)
        data = results['results']
        if data:
            df = pd.DataFrame(data)
            if set(['decimalLongitude', 'decimalLatitude', 'eventDate']).issubset(df.columns):
                frames.append(df[['decimalLongitude', 'decimalLatitude', 'eventDate']].rename(
                    columns={'decimalLongitude':'lon', 'decimalLatitude':'lat', 'eventDate':'time'}
                ))
        time.sleep(1)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['lon', 'lat', 'time'])
