import numpy as np

# Coordinates (Lat, Lon) and Time Zone Offset (Standard Time)
NBA_LOCATIONS = {
    'ATL': {'tz': -5, 'lat': 33.7490, 'lon': -84.3880},
    'BOS': {'tz': -5, 'lat': 42.3601, 'lon': -71.0589},
    'BKN': {'tz': -5, 'lat': 40.6826, 'lon': -73.9754},
    'CHA': {'tz': -5, 'lat': 35.2271, 'lon': -80.8431},
    'CHI': {'tz': -6, 'lat': 41.8781, 'lon': -87.6298},
    'CLE': {'tz': -5, 'lat': 41.4993, 'lon': -81.6944},
    'DAL': {'tz': -6, 'lat': 32.7767, 'lon': -96.7970},
    'DEN': {'tz': -7, 'lat': 39.7392, 'lon': -104.9903},
    'DET': {'tz': -5, 'lat': 42.3314, 'lon': -83.0458},
    'GSW': {'tz': -8, 'lat': 37.7749, 'lon': -122.4194},
    'HOU': {'tz': -6, 'lat': 29.7604, 'lon': -95.3698},
    'IND': {'tz': -5, 'lat': 39.7684, 'lon': -86.1581},
    'LAC': {'tz': -8, 'lat': 34.0522, 'lon': -118.2437},
    'LAL': {'tz': -8, 'lat': 34.0522, 'lon': -118.2437},
    'MEM': {'tz': -6, 'lat': 35.1495, 'lon': -90.0490},
    'MIA': {'tz': -5, 'lat': 25.7617, 'lon': -80.1918},
    'MIL': {'tz': -6, 'lat': 43.0389, 'lon': -87.9065},
    'MIN': {'tz': -6, 'lat': 44.9778, 'lon': -93.2650},
    'NOP': {'tz': -6, 'lat': 29.9511, 'lon': -90.0715},
    'NYK': {'tz': -5, 'lat': 40.7128, 'lon': -74.0060},
    'OKC': {'tz': -6, 'lat': 35.4676, 'lon': -97.5164},
    'ORL': {'tz': -5, 'lat': 28.5383, 'lon': -81.3792},
    'PHI': {'tz': -5, 'lat': 39.9526, 'lon': -75.1652},
    'PHX': {'tz': -7, 'lat': 33.4484, 'lon': -112.0740}, 
    'POR': {'tz': -8, 'lat': 45.5152, 'lon': -122.6784},
    'SAC': {'tz': -8, 'lat': 38.5816, 'lon': -121.4944},
    'SAS': {'tz': -6, 'lat': 29.4241, 'lon': -98.4936},
    'TOR': {'tz': -5, 'lat': 43.6510, 'lon': -79.3470},
    'UTA': {'tz': -7, 'lat': 40.7608, 'lon': -111.8910},
    'WAS': {'tz': -5, 'lat': 38.9072, 'lon': -77.0369},
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates flight distance in miles between two coordinates."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c