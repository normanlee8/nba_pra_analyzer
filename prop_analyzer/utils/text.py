import re
from unidecode import unidecode
from rapidfuzz import process, fuzz

def preprocess_name_for_fuzzy_match(name):
    """
    Normalizes player names for better fuzzy matching.
    Removes accents, suffixes (Jr, III), and punctuation.
    """
    if not isinstance(name, str): return ""
    
    # 1. Transliterate to ASCII (Dončić -> Doncic) and Lowercase
    name = unidecode(name).lower()
    
    # 2. Remove common suffixes
    # Matches " jr", " sr", " iii", " ii", " iv" at end of string or word boundary
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)
    
    # 3. Remove special chars (punctuation)
    name = re.sub(r'[^\w\s]', '', name)
    
    # 4. Collapse whitespace
    return " ".join(name.split())

def fuzzy_match_player(input_name, player_df, threshold=85):
    """
    Finds the best match for a player name in the master dataframe.
    Returns the player row (Series) or None.
    """
    if player_df is None or player_df.empty:
        return None
        
    clean_input = preprocess_name_for_fuzzy_match(input_name)
    
    # Use the pre-processed column if available for speed
    choices = player_df['processed_name'].tolist() if 'processed_name' in player_df.columns else player_df['clean_name'].tolist()
    
    match = process.extractOne(clean_input, choices, scorer=fuzz.token_sort_ratio)
    
    if match and match[1] >= threshold:
        # Find the row corresponding to the matched name
        # Note: We map back to the original dataframe index
        matched_name = match[0]
        # We need to find which index this corresponds to. 
        # If we used a list, we rely on position or string match.
        # Safer to filter df:
        col = 'processed_name' if 'processed_name' in player_df.columns else 'clean_name'
        row = player_df[player_df[col] == matched_name].iloc[0]
        return row
        
    return None