import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
client = OpenAI()

# Path to your vault
VAULT_PATH = "../vault"
NOTE_NAME = "test.md"  # Change to any filename inside /vault

def summarize_note():
    file_path = os.path.join(VAULT_PATH, NOTE_NAME)

    # Read the note content
    with open('/Users/pranavojha/ai-second-brain/vault/test.md', "r") as f:
        content = f.read()

    # Generate summary using GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes notes."},
            {"role": "user", "content": f"Summarize this note:\n\n{content}"}
        ]
    )

    summary = response.choices[0].message.content

    # Add summary to top of note
    new_content = f"## AI Summary\n{summary}\n\n---\n\n{content}"

    # Save the new content back to the same file
    with open('/Users/pranavojha/ai-second-brain/vault/test.md', "w") as f:
        f.write(new_content)

    print("✅ Summary added to note!")

if __name__ == "__main__":
    summarize_note()
