import json
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def visualize_chat_history(json_file_path, output_gexf_path):
    """
    Analyzes a ChatGPT conversation history JSON file and creates a GEXF graph file
    for visualization in Gephi.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    conversation_texts = []
    for conv in conversations:
        full_text = []
        if 'mapping' in conv:
            for _, message_data in sorted(conv['mapping'].items()):
                message = message_data.get('message')
                if (
                    message
                    and message.get('content')
                    and message['content'].get('content_type') == 'text'
                ):
                    parts = message['content'].get('parts', [])
                    if parts and isinstance(parts[0], str) and parts[0].strip():
                        author_role = message.get('author', {}).get('role', 'unknown')
                        full_text.append(f"{author_role.capitalize()}: {parts[0]}")
        if full_text:
            conversation_texts.append("\n".join(full_text))

    print(f"Found {len(conversation_texts)} conversations to process.")

    print("Generating embeddings for each conversation...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # OPTIONAL IMPROVEMENT: Process in batches for better memory efficiency.
    # The progress bar is re-enabled as it works well in most terminals.
    embeddings = model.encode(
        conversation_texts, 
        batch_size=32,  # Process 32 conversations at a time
        show_progress_bar=True
    )
    
    print("Embeddings generated successfully.")

    print("Calculating cosine similarity between conversations...")
    similarity_matrix = cosine_similarity(embeddings)

    G = nx.Graph()

    for i, text in enumerate(conversation_texts):
        G.add_node(i, label=f"Conversation {i}", full_text=text)

    similarity_threshold = 0.5
    for i in range(len(conversation_texts)):
        for j in range(i + 1, len(conversation_texts)):
            if similarity_matrix[i][j] > similarity_threshold:
                G.add_edge(i, j, weight=float(similarity_matrix[i][j]))

    nx.write_gexf(G, output_gexf_path)
    print(f"Graph saved to '{output_gexf_path}'. You can now open this file in Gephi.")


if __name__ == "__main__":
    visualize_chat_history('conversations.json', 'chat_history.gexf')