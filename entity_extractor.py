import spacy
from brain_assessor import assess_and_style, analyze_brain_profile_from_chat

# Load spaCy NER model
nlp = spacy.load("en_core_web_trf")  # Or use "en_core_web_trf" for better accuracy

def extract_entities_from_chat_history(chat_history):
    """Extract named entities from all user messages in chat history using spaCy."""
    user_messages_for_ner = []
    for message in chat_history:
        if message["role"] == "user":
            user_messages_for_ner.append(message["parts"][0]["text"])

    if user_messages_for_ner:
        full_user_text = "\n".join(user_messages_for_ner)
        doc = nlp(full_user_text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })
        return entities
    return []

def print_entities(entities):
    """Print extracted entities in a formatted way."""
    print("\n--- Named Entities from ALL User Inputs in History: ---")
    if entities:
        for entity in entities:
            print(f"Text: {entity['text']}, Label: {entity['label']}")
    else:
        print("No named entities found in the user inputs of the conversation.")

def print_chat_history(chat_history):
    """Print the full chat history."""
    print("\n--- Full Chat History: ---")
    for message in chat_history:
        role = message['role'].capitalize()
        text = message['parts'][0]['text']
        print(f"{role}: {text}")


