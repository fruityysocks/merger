import re
import os
from dotenv import load_dotenv
import openai
import json
from datetime import datetime
import pytz
import pandas as pd
import time
import csv

load_dotenv()
openaiApiKey = os.getenv("OPENAI_API_KEY")
if openaiApiKey is None:
    raise ValueError("OpenAI API key not found.")
openai.api_key = openaiApiKey

# Step 1: Define constants & filler words (simplified to typical fillers)
FILLERS = {'A', 'AH', 'AM', 'AN', 'AND', 'ARE',' AREN\'T', 'AS', 'AT', 'AW', 'BECAUSE', 'BUT', 'BEEN', 'COULD', 'COULDN\'T',
           'EH', 'FOR', 'FROM', 'GET', 'GONNA', 'GOT', 'GOTTA', 'GOTTEN', 'HAD', 'HAS', 'HAVE', 'HE', 'HE\'D', 'HE\'LL', 'HER',
           'HERS', 'HE\'S', 'HIS', 'HOW', 'HOW\'S', 'HUH', 'I', 'I\'LL', 'I\'M', 'I\'VE', 'I\'D', 'IN', 'IS', 'IT', 'IT\'S', 'ITS', 'JUST',
           'MY', 'NAH', 'NOT', 'OF', 'OH', 'ON', 'OR', 'OUR', 'OURS', 'SAYS', 'SHE', 'SHE\'D', 'SHE\'LL', 'SHE\'S', 'SHOULD', 'SHOULDN\'T', 'SO', 'THAN',
           'THAT', 'THAT\'S', 'THE', 'THEM', 'THERE', 'THERE\'S', 'THEY', 'THEY\'D', 'THEY\'LL', 'THEY\'RE', 'THEY\'VE', 'THEIR', 'THEIRS',
           'TO', 'UH', 'UM', 'WAS', 'WASN\'T', 'WE', 'WE\'D', 'WE\'LL', 'WERE', 'WE\'RE', 'WE\'VE', 'WHAT', 'WHEN', 'WHEN\'S', 'WHERE', 'WHERE\'S', 'WHICH', 'WHICH\'S' 
           'WHO', 'WITH', 'WOULD', 'WOULDN\'T', 'YEAH', 'YOU', 'YOURS', 'YOU\'D', 'YOU\'LL', 'YOU\'VE'}


FILLERS = {filler.lower() for filler in FILLERS}

# Step 2: Load CMUDict into a dictionary
def load_cmudict(filepath='data/cmudict-0.7b'):
    pronunciation_dict = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'):
                continue
            parts = line.strip().split('  ')
            if len(parts) == 2:
                word, phonemes = parts
                pronunciation_dict.setdefault(word.lower(), []).append(phonemes)
    return pronunciation_dict

# Step 3: Function to remove fillers from text
def remove_fillers(text):
    text = text.lower()
    for filler in FILLERS:
        text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 4: Convert transcript to phonemes using cmudict
def lookup_phonemes(word, cmu_dict):
    cleaned_word = re.sub(r'[^a-z]', '', word.lower())
    phonemes = cmu_dict.get(cleaned_word)
    if phonemes:
        return phonemes[0]
    cleaned_word_strip_digits = re.sub(r'\d+$', '', cleaned_word)
    phonemes = cmu_dict.get(cleaned_word_strip_digits)
    if phonemes:
        return phonemes[0]
    return '[UNK]'

def transcript_to_phonemes(text, cmu_dict, speaker):
    result = []
    for word in text.split():
        phonemes = lookup_phonemes(word, cmu_dict)
        result.append(f"{phonemes}")
    return result

# Step 5: Generate conversation from GPT with demographic & vocab context
def generateConversation(speaker1, speaker2, turns):
    prompt = f"""x
You are simulating a conversation between two neighbors, Speaker 1 and Speaker 2.
Speaker 1:  
- Gender: {speaker1.get('gender', 'Unknown')}  
- Location: {speaker1.get('location', 'Unknown')}  
- Profession: {speaker1.get('occupation_type', 'Unknown')}  
- Vocabulary influenced by hometown and background, but use standard English spelling.
Speaker 2:  
- Gender: {speaker2.get('gender', 'Unknown')}  
- Location: {speaker2.get('location', 'Unknown')}  
- Profession: {speaker2.get('occupation_type', 'Unknown')}  
- Vocabulary influenced by hometown and background, but use standard English spelling.
The conversation begins with casual small talk about neighborhood and everyday topics.  
After a few turns, the conversation naturally transitions to discussing their professions.
Write a conversation of {turns} alternating turns (each speaker contributing), showing this natural flow.
The conversation should be realistic, with each speaker responding to the other's comments and questions, 
also allow the agents to interrupt each other occasionally and overlap at times during their conversation.

Speaker 1: <utterance>  
Speaker 2: <utterance>  
...
Begin:
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You generate natural neighbor conversations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()

# Step 6: Extracting vowels. 
VOWELS = ['OW', 'AW', 'AA', 'AE', 'EH', 'IY', 'UW', 'AH', 'AO', 'AY', 'EY', 'IH', 'OY', 'UH']
stress_pattern = re.compile(r'^(' + '|'.join(VOWELS) + r')[12]$')

def contains_stressed_vowel(phoneme_seq):
    for phoneme in phoneme_seq:
        if stress_pattern.match(phoneme):
            return True
    return False

def extract_vowel_from_phonemes(phoneme_seq):
    for phoneme in phoneme_seq:
        if stress_pattern.match(phoneme):
            return phoneme[:-1]  
    return ''

def extract_vowel_stress(phoneme_seq):
    for phoneme in phoneme_seq:
        if stress_pattern.match(phoneme):
            return phoneme[-1] 
    return '' 

def find_vowel_index(phonemes, vowel):
    for i, p in enumerate(phonemes):
        p_base = re.sub(r'\d$', '', p)
        if p_base == vowel:
            return i
    return None

def save_stressed_tokens(phonetic_turns, cleaned_text, spk1_id, spk2_id, filename, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    turn_texts = re.split(r'\n(?=Speaker \d:)', cleaned_text)
    
    write_header = not os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['vowel', 'stress', 'preceding_segment', 'following_segment', 'word', 'phonetic_transcription', 'speaker',
                      'speaker1_id', 'speaker2_id', 'file_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        for turn_text, phonetic_turn in zip(turn_texts, phonetic_turns):
            speaker_turn = phonetic_turn['speaker'].strip() 
            speaker_num = 1 if speaker_turn == "Speaker 1:" else 2
            
            print(f"Speaker turn raw value: '{phonetic_turn['speaker']}'")
            print(f"Speaker turn stripped: '{speaker_turn}'")
            print("Speaker 1:" == speaker_turn)
            print(speaker_num)

            m = re.match(r"Speaker \d:\s*(.*)", turn_text, re.DOTALL)
            if not m:
                continue
            utterance = m.group(1).strip()
            words = utterance.split()
            
            phoneme_list = phonetic_turn['phonemes']
            
            if len(words) != len(phoneme_list):
                print(f"Warning: Mismatch between words and phonemes in {speaker_turn}")
                continue
            
            for word, phonemes_str in zip(words, phoneme_list):
                phonemes = phonemes_str.split()
                
                if contains_stressed_vowel(phonemes):
                    vowel = extract_vowel_from_phonemes(phonemes)  
                    stress = extract_vowel_stress(phonemes)  
                    index = find_vowel_index(phonemes, vowel)

                    preceding_segment = phonemes[index - 1] if index is not None and index > 0 else ''
                    following_segment = phonemes[index + 1] if index is not None and index < len(phonemes) - 1 else ''
                    
                    writer.writerow({
                        'vowel': vowel,
                        'stress': stress,
                        'preceding_segment': preceding_segment,
                        'following_segment': following_segment,
                        'word': word,
                        'phonetic_transcription': phonemes_str,
                        'speaker': speaker_num,
                        'speaker1_id': spk1_id,
                        'speaker2_id': spk2_id,
                        'file_name': filename,
                    })
                    print(f"Extracted vowel '{vowel}' with preceding segment '{preceding_segment}' and following segment '{following_segment}' "
                          f"from word '{word}' and speaker1 '{spk1_id}' and speaker2 '{spk2_id}'.")
                    
# Step 7: Function to save conversation to a JSON file 
def save_conversation_json(speaker1, speaker2, original, cleaned, phonetic, outdir="conversations"):
    os.makedirs(outdir, exist_ok=True)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{speaker1}_and_{speaker2}_{timestamp}.json"
    filename = f"{speaker1}_and_{speaker2}.json"
    filepath = os.path.join(outdir, filename)
    
    data = {
        "speaker1": speaker1,
        "speaker2": speaker2,
        "original_conversation": original,
        "cleaned_conversation": cleaned,
        "phonetic_transcription": phonetic
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved conversation JSON to {filepath}")
    
# Step 8: Main function
def main():
    speakers_df = pd.read_csv('data/demographicInfo.csv')
    speakers_df['location'] = speakers_df['childcity'] + ", " + speakers_df['childstate']
    neighbors_df = pd.read_csv('data/neighborPairs.csv')  
    cmu_dict = load_cmudict('data/cmudict-0.7b.txt')
    speaker_info_dict = speakers_df.set_index('speaker').to_dict(orient='index')
    
    for idx, row in neighbors_df.iterrows():
        if idx > 20: 
            break
        spk1_id = row['speaker1']
        spk2_id = row['speaker2']
        speaker1 = speaker_info_dict.get(spk1_id)
        speaker2 = speaker_info_dict.get(spk2_id)
        
        if not speaker1 or not speaker2:
            print(f"Skipping pair with missing speaker info: {spk1_id}, {spk2_id}")
            continue
        conversation_text = generateConversation(speaker1, speaker2, turns=20)
        print(f"\nOriginal conversation between {spk1_id} and {spk2_id}:\n{conversation_text}\n")
        turn_texts = re.split(r'\n(?=Speaker \d:)', conversation_text)
        cleaned_turns = []
        phonetic_turns = []
        
        for turn in turn_texts:
            match = re.match(r'(Speaker \d:)\s*(.*)', turn, re.DOTALL)
            if not match:
                continue
            speaker_tag, utterance = match.groups()
            cleaned_utterance = remove_fillers(utterance)
            cleaned_turns.append(f"{speaker_tag} {cleaned_utterance}")
            if speaker_tag.strip() == "Speaker 1:":
                spk = spk1_id
            else:
                spk = spk2_id
            phonemes = transcript_to_phonemes(cleaned_utterance, cmu_dict, spk)
            phonetic_turns.append({"speaker": speaker_tag, "phonemes": phonemes})
        
        cleaned_text = "\n".join(cleaned_turns)
        print(f"Cleaned conversation:\n{cleaned_text}\n")
        
        for pturn in phonetic_turns:
            print(f"{pturn['speaker']} phonemes: {pturn['phonemes']}")
    
        save_conversation_json(
            speaker1=spk1_id,
            speaker2=spk2_id,
            original=conversation_text,
            cleaned=cleaned_text,
            phonetic=phonetic_turns,
            outdir="conversations/day3"
        )
        
        save_stressed_tokens(
            phonetic_turns, cleaned_text, spk1_id, spk2_id,
            filename=f"{spk1_id}_and_{spk2_id}.json",
            output_csv="data/day3Vowels.csv"
        )
        
        time.sleep(1.5)

if __name__ == "__main__":
    main()