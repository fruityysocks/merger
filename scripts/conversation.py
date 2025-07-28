import os
from dotenv import load_dotenv
import openai
import json
from datetime import datetime
import pytz

load_dotenv()

openaiApiKey = os.getenv("OPENAI_API_KEY")
if openaiApiKey is None:
    raise ValueError("OpenAI API key not found.")
openai.api_key = openaiApiKey

filename = f"conversations/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
eastern = pytz.timezone('America/New_York')

def generateChatCompletion(messages, model="gpt-4", temperature=0.7, max_tokens=150):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def runConversation(topic, turns, speaker1Profile, speaker2Profile):
    system_content = (
        f"You are simulating a conversation between two speakers.\n"
        f"Speaker 1 profile: {speaker1Profile}\n"
        f"Speaker 2 profile: {speaker2Profile}\n"
        f"Start a discussion on the topic: {topic}"
    )
    messages = [{"role": "system", "content": system_content}]
    
    transcript = []
    
    messages.append({"role": "user", "content": topic})
    now = datetime.now(eastern).isoformat()
    transcript.append({"speaker": "Speaker 1", "text": topic, "timestamp": now})
    
    reply = generateChatCompletion(messages)
    now = datetime.now(eastern).isoformat()
    transcript.append({"speaker": "Speaker 2", "text": reply, "timestamp": now})
    messages.append({"role": "assistant", "content": reply})

    for i in range(turns):
        reply1 = generateChatCompletion(messages + [{"role": "user", "content": f"Speaker 1:"}])
        
        now = datetime.now(eastern).isoformat()
        transcript.append({"speaker": "Speaker 1", "text": reply1, "timestamp": now})
        messages.append({"role": "user", "content": reply1})

        reply2 = generateChatCompletion(messages + [{"role": "user", "content": f"Speaker 2:"}])
        now = datetime.now(eastern).isoformat()
        transcript.append({"speaker": "Speaker 2", "text": reply2, "timestamp": now})
        messages.append({"role": "assistant", "content": reply2})

    return transcript


if __name__ == "__main__":
    speaker1Profile = input("Enter profile for Speaker 1: ")
    speaker2Profile = input("Enter profile for Speaker 2: ")
    startingTopic = input("Enter starting topic: ")
    turns = int(input("Enter number of turns: "))    

    conversationTranscript = runConversation(
        topic=startingTopic, 
        turns=turns, 
        speaker1Profile=speaker1Profile, 
        speaker2Profile=speaker2Profile
    )

    for turn in conversationTranscript:
        print(f"{turn['speaker']}: {turn['text']}\n")

    os.makedirs("conversations", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(conversationTranscript, f, indent=2)