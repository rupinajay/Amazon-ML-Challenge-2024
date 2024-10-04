import pandas as pd
import re

# Define the entity_unit_map
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Define a unit conversion mapping, including the milliwatt to millivolt change
unit_conversion = {
    'cm': 'centimetre',
    'mm': 'millimetre',
    'm': 'metre',
    'kg': 'kilogram',
    'g': 'gram',
    'mg': 'milligram',
    'μg': 'microgram',
    'lb': 'pound',
    'lbs': 'pound',
    'oz': 'ounce',
    't': 'ton',
    'kv': 'kilovolt',
    'mv': 'millivolt',  # Corrected to millivolt
    'v': 'volt',
    'kw': 'kilowatt',
    'w': 'watt',
    'ml': 'millilitre',
    'l': 'litre',
    'cl': 'centilitre',
    'dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'imp gal': 'imperial gallon',
    'pt': 'pint',
    'qt': 'quart',
    'cu ft': 'cubic foot',
    'cu in': 'cubic inch',
    'milliwatt': 'millivolt'
}

# Load the CSV file
df = pd.read_csv('FINALFILE.csv')

# Function to check if unit is valid
def is_valid_unit(entity_name, unit):
    if unit is None:
        return False
    unit = unit.lower()
    valid_units = entity_unit_map.get(entity_name, {})
    return unit in valid_units

# Function to clear invalid units and format values
def clear_invalid_units(row):
    entity_name = row['entity_name']
    entity_value = row['entity_value']
    
    if pd.notna(entity_value):
        # Convert values starting with '.' to '0.'
        if entity_value.startswith('.'):
            entity_value = '0' + entity_value
        
        match = re.match(r'(\d*\.?\d*)\s*([a-zA-Zµ]+)?', entity_value)
        if match:
            value, unit = match.groups()
            unit = unit.lower() if unit else None
            if not is_valid_unit(entity_name, unit):
                return ""
            # Return formatted value
            return f"{value} {unit}" if unit else value
    
    return entity_value

# Apply the function to clear invalid units and format values
df['entity_value'] = df.apply(clear_invalid_units, axis=1)

# Save the cleaned DataFrame with only 'index' and 'entity_value'
df[['index', 'entity_value']].to_csv('CLEANED_FINALFILE.csv', index=False)

print("Processing complete. The cleaned file is saved as 'CLEANED_FINALFILE.csv'.")
