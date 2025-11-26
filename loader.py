def get_all_local_transcripts_from_file(filename):
    """Reads and returns text from a single local text file."""
    print(f"\n--- Reading knowledge base from '{filename}' ---")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully read '{filename}'.")
            return content
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{filename}' not found. Please ensure it's in the same directory.")
        return ""
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return ""

def load_all_knowledge_bases(file_paths):
    """Load and combine content from multiple knowledge base files."""
    contents = []
    for file_path in file_paths:
        content = get_all_local_transcripts_from_file(file_path)
        if content:
            contents.append(content)
    
    combined_content = "\n\n".join(contents)
    return combined_content