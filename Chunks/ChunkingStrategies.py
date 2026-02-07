from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import re
from dataclasses import dataclass


# Fixed Size chunking with Overlap
def fixed_size_chunking(text: str, source: str, chunk_size: int = 700, overlap: int = 100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunks.append({
            "text": " ".join(words[i:i + chunk_size]),
            "source": source
        })

    return chunks

# Sentence-based chunking
def sentence_based_chunking(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text into chunks of N sentences.

    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk

    Returns:
        List of sentence-based chunks
    """
    # Simple sentence splitter (improved version would handle edge cases)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]  # Remove empty

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks

# Recursive chunking
def recursive_chunking(text: str, max_chunk_size: int = 500, separators: List[str] = None) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.

    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
        separators: List of separators in order of preference

    Returns:
        List of recursively split chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]  # Paragraph, line, sentence, word

    def split_recursive(text: str, seps: List[str]) -> List[str]:
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []

        if not seps:
            # No more separators, do character split
            return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        separator = seps[0]
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= max_chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(split) > max_chunk_size:
                    # Split is too large, recurse with next separator
                    chunks.extend(split_recursive(split, seps[1:]))
                    current_chunk = ""
                else:
                    current_chunk = split + separator

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    return split_recursive(text, separators)

# Semantic chunking
def semantic_chunking(text: str, model: SentenceTransformer, similarity_threshold: float = 0.5) -> List[str]:
    """
    Split text at points where semantic similarity drops below threshold.

    Args:
        text: Input text
        model: Sentence transformer model
        similarity_threshold: Cosine similarity threshold for splitting

    Returns:
        List of semantically coherent chunks
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    if len(sentences) <= 1:
        return sentences

    # Encode all sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        similarities.append(sim)

    # Find split points where similarity drops
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            # Split here - low similarity means topic change
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Structure-Aware chunking
@dataclass
class Section:
    """Represents a document section with hierarchy."""
    level: int  # Header level (1 for #, 2 for ##, etc.)
    title: str
    content: str
    parent: Optional['Section'] = None
    children: List['Section'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

def structure_aware_chunking(text: str) -> List[Section]:
    """
    Parse markdown structure into hierarchical sections.

    Args:
        text: Markdown-formatted text

    Returns:
        List of Section objects with hierarchy
    """
    lines = text.strip().split('\n')
    sections = []
    current_section = None
    current_content = []

    for line in lines:
        # Check for markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            # Save previous section
            if current_section is not None:
                current_section.content = '\n'.join(current_content).strip()
                sections.append(current_section)

            # Create new section
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_section = Section(level=level, title=title, content="")
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section is not None:
        current_section.content = '\n'.join(current_content).strip()
        sections.append(current_section)

    return sections
