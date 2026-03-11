import numpy as np
import pytz
import pandas as pd

# Coordinates (Lat, Lon) and IANA Time Zone Strings (Handles DST dynamically)
NBA_LOCATIONS = {
    'ATL': {'tz': 'America/New_York', 'lat': 33.7490, 'lon': -84.3880},
    'BOS': {'tz': 'America/New_York', 'lat': 42.3601, 'lon': -71.0589},
    'BKN': {'tz': 'America/New_York', 'lat': 40.6826, 'lon': -73.9754},
    'CHA': {'tz': 'America/New_York', 'lat': 35.2271, 'lon': -80.8431},
    'CHI': {'tz': 'America/Chicago', 'lat': 41.8781, 'lon': -87.6298},
    'CLE': {'tz': 'America/New_York', 'lat': 41.4993, 'lon': -81.6944},
    'DAL': {'tz': 'America/Chicago', 'lat': 32.7767, 'lon': -96.7970},
    'DEN': {'tz': 'America/Denver', 'lat': 39.7392, 'lon': -104.9903},
    'DET': {'tz': 'America/Detroit', 'lat': 42.3314, 'lon': -83.0458},
    'GSW': {'tz': 'America/Los_Angeles', 'lat': 37.7749, 'lon': -122.4194},
    'HOU': {'tz': 'America/Chicago', 'lat': 29.7604, 'lon': -95.3698},
    'IND': {'tz': 'America/Indiana/Indianapolis', 'lat': 39.7684, 'lon': -86.1581},
    'LAC': {'tz': 'America/Los_Angeles', 'lat': 34.0522, 'lon': -118.2437},
    'LAL': {'tz': 'America/Los_Angeles', 'lat': 34.0522, 'lon': -118.2437},
    'MEM': {'tz': 'America/Chicago', 'lat': 35.1495, 'lon': -90.0490},
    'MIA': {'tz': 'America/New_York', 'lat': 25.7617, 'lon': -80.1918},
    'MIL': {'tz': 'America/Chicago', 'lat': 43.0389, 'lon': -87.9065},
    'MIN': {'tz': 'America/Chicago', 'lat': 44.9778, 'lon': -93.2650},
    'NOP': {'tz': 'America/Chicago', 'lat': 29.9511, 'lon': -90.0715},
    'NYK': {'tz': 'America/New_York', 'lat': 40.7128, 'lon': -74.0060},
    'OKC': {'tz': 'America/Chicago', 'lat': 35.4676, 'lon': -97.5164},
    'ORL': {'tz': 'America/New_York', 'lat': 28.5383, 'lon': -81.3792},
    'PHI': {'tz': 'America/New_York', 'lat': 39.9526, 'lon': -75.1652},
    'PHX': {'tz': 'America/Phoenix', 'lat': 33.4484, 'lon': -112.0740}, 
    'POR': {'tz': 'America/Los_Angeles', 'lat': 45.5152, 'lon': -122.6784},
    'SAC': {'tz': 'America/Los_Angeles', 'lat': 38.5816, 'lon': -121.4944},
    'SAS': {'tz': 'America/Chicago', 'lat': 29.4241, 'lon': -98.4936},
    'TOR': {'tz': 'America/Toronto', 'lat': 43.6510, 'lon': -79.3470},
    'UTA': {'tz': 'America/Denver', 'lat': 40.7608, 'lon': -111.8910},
    'WAS': {'tz': 'America/New_York', 'lat': 38.9072, 'lon': -77.0369},
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

def get_tz_shift(loc1, loc2, date):
    """
    Calculates the exact timezone shift considering DST for a given date.
    Positive means traveling East (lose sleep), Negative means West.
    """
    if pd.isna(date):
        date = pd.Timestamp.today()
        
    tz1 = pytz.timezone(NBA_LOCATIONS[loc1]['tz'])
    tz2 = pytz.timezone(NBA_LOCATIONS[loc2]['tz'])
    
    # Localize a standard noon time to extract the specific UTC offset on that day
    dt1 = tz1.localize(date.replace(hour=12, minute=0, second=0))
    dt2 = tz2.localize(date.replace(hour=12, minute=0, second=0))
    
    offset1_hours = dt1.utcoffset().total_seconds() / 3600
    offset2_hours = dt2.utcoffset().total_seconds() / 3600
    
    return offset2_hours - offset1_hours