import os
import google.generativeai as genai
from mem0 import Memory

memory = Memory()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

GROUP_ID = "group_chat_1"
PARTICIPANTS = {
    "archie": "Archie",
    "arjun": "Arjun",
    "bheem": "Bheem"
}

def print_separator(title=""):
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{'-' * padding} {title} {'-' * padding}\n")
    else:
        print(f"\n{'-' * width}\n")

def process_message(participant, message):
    memory.add(
        message,
        group_id=GROUP_ID,
        metadata={"participant": participant}
    )
    
    relevant_memories = memory.search(
        query=message,
        group_id=GROUP_ID,
    )
    
    memories_str = ""
    if relevant_memories["results"]:
        memories_str = "Previous memories from the conversation:\n" + "\n".join(
            f"- {m['memory']} (from: {m.get('metadata', {}).get('participant', 'unknown')})" 
            for m in relevant_memories["results"]
        )
    
    system_prompt = f"""You are a friendly and engaging assistant in a group chat. Your role is to acknowledge and respond to messages in a warm, conversational manner.

You are currently talking to {PARTICIPANTS[participant]}.
{memories_str}

Rules for your response:
1. Keep responses warm and friendly, using emojis naturally
2. Reference previous context when relevant
3. Keep responses brief but engaging
4. Acknowledge specific details mentioned in the message
5. Show enthusiasm about shared interests and plans
6. If you don't have enough context, give a friendly acknowledgment
7. Don't make assumptions beyond what's explicitly shared
8. Don't suggest new plans unless they were mentioned

Current message from {PARTICIPANTS[participant]}: {message}"""
    
    response = model.generate_content([
        {"role": "user", "parts": [system_prompt]}
    ])
    
    assistant_response = response.text
    
    memory.add(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_response}
        ],
        group_id=GROUP_ID,
        metadata={"participant": participant}
    )
    
    return assistant_response

def display_memories(participant=None):
    title = f"Memories for {PARTICIPANTS[participant]}" if participant else "All Group Memories"
    print_separator(title)
    
    memories = memory.search(
        query="all memories",
        group_id=GROUP_ID,
    )
    
    if participant:
        memories["results"] = [
            mem for mem in memories["results"]
            if mem.get("metadata", {}).get("participant") == participant
        ]
    
    if not memories["results"]:
        print("No memories found.")
        return
    
    memories["results"].sort(key=lambda x: x.get("score", 0), reverse=True)
    
    for mem in memories["results"]:
        participant = mem.get("metadata", {}).get("participant", "unknown")
        print(f"[{participant}] {mem['memory']}")
    
    if participant:
        print(f"\n----------------- Memories for {PARTICIPANTS[participant]} (including group context) -----------------")
        group_memories = memory.search(
            query="all memories",
            group_id=GROUP_ID,
        )
        group_memories["results"].sort(key=lambda x: x.get("score", 0), reverse=True)
        
        print("\n--- Personal Memories ---")
        for mem in group_memories["results"]:
            mem_participant = mem.get("metadata", {}).get("participant", "unknown")
            if mem_participant == participant:
                print(f"[{mem_participant}] {mem['memory']}")
        
        print("\n--- Group Context ---")
        for mem in group_memories["results"]:
            mem_participant = mem.get("metadata", {}).get("participant", "unknown")
            if mem_participant != participant:
                print(f"[{mem_participant}] {mem['memory']}")

def simulate_group_chat():
    print_separator("Group Chat Simulation")
    
    conversation = [
        {"participant": "archie", "message": "Hi everyone! I'm Archie. I love classical dance and reading mythological books."},
        {"participant": "arjun", "message": "Hey there, I'm Arjun. I'm really into cricket and food photography."},
        {"participant": "bheem", "message": "Hello! Bheem here. I enjoy playing tabla and yoga."},
        
        {"participant": "archie", "message": "What should we do this weekend? I was thinking maybe a trip to Elephanta Caves?"},
        {"participant": "arjun", "message": "The caves sound amazing! I could bring my camera and take some heritage photos. Maybe I can also pack us some homemade biryani?"},
        {"participant": "bheem", "message": "I'm definitely in for the caves. The weather forecast says it's going to be pleasant at 24°C."},
        
        {"participant": "archie", "message": "Perfect! Let's meet at the Gateway of India at 9am on Saturday. The ferry ride is about 1 hour."},
        {"participant": "arjun", "message": "9am works for me. I'll prepare some biryani and snacks. Any food preferences or allergies I should know about?"},
        {"participant": "bheem", "message": "I'm vegetarian, but otherwise I eat anything. I can bring some lassi and nimbu pani for everyone."},
        
        {"participant": "archie", "message": "Great! So we're set for Saturday at 9am. I'll bring a first aid kit and guide book."},
        {"participant": "arjun", "message": "I'll make sure to pack both veg and non-veg options. Looking forward to our trip!"},
        {"participant": "bheem", "message": "Sounds like a plan! I'll also bring some masala chai in a thermos. See you all on Saturday!"}
    ]
    
    for turn in conversation:
        participant = turn["participant"]
        message = turn["message"]
        
        print(f"\n{PARTICIPANTS[participant]}: {message}")
        response = process_message(participant, message)
        print(f"Assistant: {response}")

def demonstrate_memory_retrieval():
    print("\nResetting all memories")
    print("Resetting index mem0...\n")
    
    display_memories(participant="archie")
    display_memories(participant="arjun")
    display_memories(participant="bheem")
    display_memories()
    
    print_separator("Memory Query Examples")
    
    query = "What does Archie like to do?"
    print(f"Query: {query}")
    archie_interests = memory.search(
        query=query,
        group_id=GROUP_ID,
        filters={"metadata.participant": "archie"},
        limit=3
    )
    for mem in archie_interests["results"]:
        print(f"- {mem['memory']} (score: {mem.get('score', 'N/A')})")
    
    print()
    
    query = "What are the weekend plans?"
    print(f"Query: {query}")
    weekend_plans = memory.search(
        query=query,
        group_id=GROUP_ID,
        limit=5
    )
    for mem in weekend_plans["results"]:
        participant = mem.get("metadata", {}).get("participant", "unknown")
        print(f"- [{participant}] {mem['memory']} (score: {mem.get('score', 'N/A')})")
    
    print()
    
    query = "Who has dietary restrictions?"
    print(f"Query: {query}")
    dietary = memory.search(
        query=query,
        group_id=GROUP_ID,
        limit=3
    )
    for mem in dietary["results"]:
        participant = mem.get("metadata", {}).get("participant", "unknown")
        print(f"- [{participant}] {mem['memory']} (score: {mem.get('score', 'N/A')})")
    
    print()
    
    query = "What activities do all participants enjoy?"
    print(f"Query: {query}")
    activities = memory.search(
        query=query,
        group_id=GROUP_ID,
        limit=9
    )
    
    participant_activities = {}
    for mem in activities["results"]:
        participant = mem.get("metadata", {}).get("participant", "unknown")
        if participant not in participant_activities:
            participant_activities[participant] = []
        participant_activities[participant].append(mem)
    
    for participant, mems in participant_activities.items():
        print(f"\n[{participant}]'s activities:")
        for mem in mems:
            print(f"- {mem['memory']} (score: {mem.get('score', 'N/A')})")
    
    print("\n-------------------------------- Demo Complete --------------------------------")

if __name__ == "__main__":
    memory.reset()
    simulate_group_chat()
    demonstrate_memory_retrieval()
