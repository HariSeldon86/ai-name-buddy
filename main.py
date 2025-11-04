import json
from rich import print

from database import (
    check_abbreviation_exists,
    check_keyword_exists,
    insert_word,
    setup_database,
)
from models import Word
from vectorstore import (
    add_word_to_vectorstore,
    get_or_create_vectorstore,
)
from agents import agent_executor
from config import Config




if __name__ == "__main__":
    setup_database()
    vectorstore = get_or_create_vectorstore()

    while True:
        user_keyword = input("\nEnter a new keyword (type 'exit' to quit): ")
        if user_keyword.lower() == "exit":
            print("Exiting agent. Goodbye!")
            break

        if check_keyword_exists(user_keyword):
            print(f"✗ The keyword '{user_keyword}' already exists in the database. Please try again.")
            continue

        try:
            response = agent_executor.invoke({"input": user_keyword})
            print("\n" + "="*80)
            print("FINAL SUGGESTION:")
            print("="*80)
            

            output_text = response["output"]
            output_json = json.loads(output_text)
            
            # Try to extract abbreviation and description from the output
            abbr = output_json.get("abbreviation")
            desc = output_json.get("description")
                
            if abbr:
                [print(f"{key}: {value}") for key, value in output_json.items()]
                print("="*80)

                if check_abbreviation_exists(abbr):
                    print(f"✗ The suggested abbreviation '{abbr}' already exists in the database. Please try again.")
                    continue

                # Ask user if they want to save this to the database
                save_choice = input("\nDo you want to add this word to the database? (yes/no): ").strip().lower()
            
                if save_choice in ['yes', 'y']:
                    word = Word(keyword=user_keyword, abbreviation=abbr, description=desc)

                    # Insert into SQLite database
                    if insert_word(word):
                        # Add to ChromaDB vector store
                        try:
                            add_word_to_vectorstore(word)
                            print("\n✓ Word successfully saved to both databases!")
                        except Exception as e:
                            print(f"✗ Error adding to vector store: {e}")
                    else:
                        print("✗ Failed to add word to database.")
                else:
                    print("Word not saved.")
            else:
                print(response["output"])
                print("✗ Could not extract abbreviation from the AI response. Please try again.")
                print("="*80)
                
        except Exception as e:
            print(f"An error occurred: {e}")
            print(
                f"Please ensure Ollama is running and the LLM model ('{Config.OLLAMA_LLM_MODEL}') is available."
            )