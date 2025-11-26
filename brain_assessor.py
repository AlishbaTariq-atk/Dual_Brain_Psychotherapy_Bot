from transformers import pipeline

# Initialize the emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    return_all_scores=False
)

# Map each emotion label to a brain hemisphere
def map_emotion_to_side(label: str) -> str:
    label = label.lower()

    right_emotions = {
        "sadness", "joy", "fear", "anger", "disgust", "surprise", "grief",
        "desire", "nervousness", "excitement", "disappointment", "remorse"
    }
    left_emotions = {
        "neutral", "confusion", "approval", "realization", "curiosity"
    }

    if label in right_emotions:
        return "right"
    elif label in left_emotions:
        return "left"
    else:
        return "balanced"

# Analyze a single message
def assess_message(message: str) -> dict:
    output = emotion_classifier(message)[0]
    label = output["label"].lower()
    score = output["score"]
    dominant = map_emotion_to_side(label)

    return {
        "dominant_side": dominant,
        "confidence": round(score, 2),
        "emotion_label": label,
        "message": message
    }

# Format output with styled prefix
def assess_and_style(message: str) -> dict:
    result = assess_message(message)
    label = result["emotion_label"].capitalize()
    conf = result["confidence"]
    side = result["dominant_side"]

    prefix = {
        "left": "[LOGICAL 🧠]",
        "right": "[EMOTIONAL ❤️]",
        "balanced": "[MIXED ⚖️]"
    }[side]

    result["styled_text"] = f'{prefix} "{message}" — ({label}, {conf})'
    return result

# Analyze an entire chat and build transition matrix
def analyze_brain_profile_from_chat(chat_history: list) -> dict:
    user_msgs = [msg["parts"][0]["text"] for msg in chat_history if msg.get("role") == "user"]

    if not user_msgs:
        return {
            "state": "unknown",
            "reason": "No user messages found.",
            "dominance_distribution": {},
            "messages": [],
            "transitions": {
                "labels": ["stressed", "mature"],
                "normalized": [[0, 0], [0, 0]],
                "raw_counts": [[0, 0], [0, 0]],
                "pretty_normalized": {},
                "pretty_raw_counts": {}
            }
        }

    assessments = []
    state_labels = ["stressed", "mature"]
    state_to_index = {state: i for i, state in enumerate(state_labels)}
    dominant_counts = {state: 0 for state in state_labels}
    raw_transitions = [[0, 0], [0, 0]]
    total_transitions = 0
    previous_state = None

    for i, text in enumerate(user_msgs, start=1):
        res = assess_and_style(text)
        side = res["dominant_side"]  # "left", "right", or "balanced"

        # Map left/right to mature/stressed
        if side == "right":
            state = "stressed"
        elif side == "left":
            state = "mature"
        else:
            state = None  # Ignore "balanced" or anything else

        # Record full message analysis
        assessments.append({
            "message_number": i,
            "original": text,
            "emotion": res["emotion_label"],
            "confidence": res["confidence"],
            "dominant_side": side,
            "dbp_state": state,
            "styled": res["styled_text"]
        })

        # Count dominance by DBP state
        if state in dominant_counts:
            dominant_counts[state] += 1

        # Track transitions (only if both current and previous states are valid)
        if previous_state in state_labels and state in state_labels:
            from_idx = state_to_index[previous_state]
            to_idx = state_to_index[state]
            raw_transitions[from_idx][to_idx] += 1
            total_transitions += 1

        if state in state_labels:
            previous_state = state

    # Normalize transition matrix
    if total_transitions == 0:
        normalized_matrix = [[0, 0], [0, 0]]
    else:
        normalized_matrix = [
            [round(cell / total_transitions, 4) for cell in row]
            for row in raw_transitions
        ]

    # Pretty labeled matrices for JSON
    pretty_normalized = {
        "from\\to": {
            from_label: {
                to_label: normalized_matrix[i][j]
                for j, to_label in enumerate(state_labels)
            }
            for i, from_label in enumerate(state_labels)
        }
    }
    pretty_raw_counts = {
        "from\\to": {
            from_label: {
                to_label: raw_transitions[i][j]
                for j, to_label in enumerate(state_labels)
            }
            for i, from_label in enumerate(state_labels)
        }
    }

    # Determine dominant DBP state
    dominant_brain = max(dominant_counts, key=dominant_counts.get)

    return {
        "state": dominant_brain,
        "reason": f"User primarily responded from the {dominant_brain.upper()} mind.",
        "dominance_distribution": dominant_counts,
        "messages": assessments,
        "transitions": {
            "labels": state_labels,
            "normalized": normalized_matrix,
            "raw_counts": raw_transitions,
            "pretty_normalized": pretty_normalized,
            "pretty_raw_counts": pretty_raw_counts
        }
    }

