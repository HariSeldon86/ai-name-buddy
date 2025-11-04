import sqlite3
import json
from typing import get_type_hints, get_args, get_origin
from rich import print

from models import Word
from config import Config


def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_PATH)
    except sqlite3.Error as e:
        print(e)
    return conn

def get_sql_type(python_type) -> str:
    """Map Python type hints to SQLite types."""
    # Handle Optional types
    origin = get_origin(python_type)
    if origin is not None:
        # For Optional[X], get the actual type X
        args = get_args(python_type)
        if args:
            python_type = args[0]
    
    # Map Python types to SQLite types
    type_mapping = {
        str: "TEXT",
        int: "INTEGER",
        float: "REAL",
        bool: "INTEGER",  # SQLite uses INTEGER for boolean
        bytes: "BLOB"
    }
    
    return type_mapping.get(python_type, "TEXT")

def create_table_from_model(conn):
    """Dynamically create the words table based on the Word model."""
    # Get field information from the Word model
    fields = Word.model_fields
    
    # Build column definitions
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    
    for field_name, field_info in fields.items():
        field_type = field_info.annotation
        sql_type = get_sql_type(field_type)
        
        # Determine if field is required (not Optional)
        is_optional = get_origin(field_type) is not None  # Optional creates Union type
        
        # Build column definition
        column_def = f"{field_name} {sql_type}"
        
        # Add constraints
        if not is_optional:
            column_def += " NOT NULL"
        
        # Add unique constraint for keyword and abbreviation
        if field_name in ["keyword", "abbreviation"]:
            column_def += " UNIQUE"
        
        columns.append(column_def)
    
    # Create the SQL statement
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS words (
            {', '.join(columns)}
        )
    """
    
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        print(f"Table 'words' created/verified with schema based on Word model.")
        print(f"Columns: {[col.split()[0] for col in columns]}")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def get_field_names():
    """Get list of field names from Word model (excluding 'id')."""
    return list(Word.model_fields.keys())

def create_table(conn):
    """Create the words table if it doesn't exist (legacy function, now uses model-based creation)."""
    create_table_from_model(conn)

def is_database_populated(conn):
    """Check if the database already has data."""
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM words")
    count = c.fetchone()[0]
    return count > 0

def populate_database(conn):
    """Populate the database from the JSON file using Word model."""
    if is_database_populated(conn):
        print("Database is already populated.")
        return

    print(f"Populating database from {Config.DICTIONARY_JSON_PATH}...")
    with open(Config.DICTIONARY_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to Word objects for validation
    words = [Word(**item) for item in data]
    
    # Get field names dynamically
    field_names = get_field_names()
    placeholders = ', '.join(['?' for _ in field_names])
    columns = ', '.join(field_names)
    
    insert_sql = f"INSERT INTO words ({columns}) VALUES ({placeholders})"
    
    c = conn.cursor()
    for word in words:
        # Get values in the same order as field_names
        values = [getattr(word, field) for field in field_names]
        c.execute(insert_sql, values)
    
    conn.commit()
    print(f"Database populated successfully with {len(words)} words.")

def check_keyword_exists(keyword: str) -> bool:
    """Check if a keyword exists in the database."""
    conn = create_connection()
    if conn is None:
        return False
    
    c = conn.cursor()
    c.execute("SELECT 1 FROM words WHERE keyword = ?", (keyword,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def check_abbreviation_exists(abbreviation: str) -> bool:
    """Check if an abbreviation exists in the database."""
    conn = create_connection()
    if conn is None:
        return False
    
    c = conn.cursor()
    c.execute("SELECT 1 FROM words WHERE abbreviation = ?", (abbreviation,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def insert_word(word: Word) -> bool:
    """Insert a new word into the database using Word model."""
    conn = create_connection()
    if conn is None:
        return False
    
    try:        
        # Get field names and values dynamically
        field_names = get_field_names()
        placeholders = ', '.join(['?' for _ in field_names])
        columns = ', '.join(field_names)
        
        insert_sql = f"INSERT INTO words ({columns}) VALUES ({placeholders})"
        values = [getattr(word, field) for field in field_names]
        
        c = conn.cursor()
        c.execute(insert_sql, values)
        conn.commit()
        conn.close()
        print(f"Added '{word.keyword}' to database.")
        return True
    except sqlite3.Error as e:
        print(f"Error inserting word: {e}")
        conn.close()
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        if conn:
            conn.close()
        return False

def get_all_words() -> list[Word]:
    """Retrieve all words from the database and return them as a list of Word objects."""
    conn = create_connection()
    if conn is None:
        return []

    words = []
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM words")
        rows = c.fetchall()
        
        # Get field names from the Word model to ensure correct mapping
        model_fields = get_field_names()
        
        for row in rows:
            # Create a dictionary from the row object
            row_dict = dict(row)
            
            # Filter out any columns from the DB that are not in the Word model
            filtered_dict = {key: row_dict[key] for key in model_fields if key in row_dict}
            
            # Create a Word object
            words.append(Word(**filtered_dict))
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            
    return words

def setup_database():
    """Orchestrates the creation and population of the database."""
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        populate_database(conn)
        conn.close()
        print("Database setup complete.")
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    setup_database()
