# imports
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from serpapi import GoogleSearch
from Chunks.ChunkingStrategies import fixed_size_chunking
from google import genai

load_dotenv()


class WebQuerySearch:
    def __init__(self):
        self.serpapi_key = os.getenv("serpapi_key")
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.gemini_model = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME")

    def get_best_result(self, query, k=3):
        # Get URLs
        url_list = self.get_url(query)

        # Extract doc from urls
        doc_list = self.extract_text_from_urls(url_list)

        # Chunk each doc
        all_chunks = []
        for url, text in doc_list:
            all_chunks.extend(self.chunk_text(url, text))

        # Embed each chunk
        query_embedding, chunk_embedding = self.embed_chunks(query, all_chunks)

        # Compare and get best chunk
        score = util.cos_sim(query_embedding, chunk_embedding)[0] # 0th index because util.cos_sim returns a 2D matrix by default e.g. [[0.827]]
        # bestIdx = int(score.argmax()) # score.argmax() gives the index of the highest matching cosine similarity
        top_indices = score.topk(k).indices.tolist() # score.topk(k) returns a named tuple (values, indices)
        return [all_chunks[i] for i in top_indices]


    def get_url(self, query):
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            }

            searched_urls = GoogleSearch(params).get_dict()
            return [r["link"] for r in searched_urls.get("organic_results", [])]
        except Exception as e:
            raise RuntimeError(f"Unable to do find relevant URLs: {e}")

    def extract_text_from_urls(self, url_list):
        try:
            doc_list = []
            for url in url_list:
                response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}) # Python’s requests.get() default User-Agent is: python-requests/2.x.x. Many websites treat this as: A bot, scraper, potential abuse vector and block the access
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = " ".join(soup.get_text(separator=" ").split())

                # Very likely paywall / blocked content
                if len(text) < 500: # E.g. "sign in to continue", "Subscribe to read more" etc
                    continue

                doc_list.append((url, text))
            return doc_list
        except Exception as e:
            raise RuntimeError(f"Unable to extract data from urls: {e}")

    def chunk_text(self, url, document):
        return fixed_size_chunking(text=document, source=url) # why fixed-size chunking?

    def embed_chunks(self, query, all_chunks):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True) # convert_to_tensor=True because "util.cos_sim" expects tensor array instead of (default) numpy array
        chunk_embedding = self.embedding_model.encode([c["text"] for c in all_chunks], convert_to_tensor=True) # encoding multiple queries at once here using "batched embedding call" to increase performance
        return query_embedding, chunk_embedding

    def main(self):
        # Take user query
        while(True):
            query = input("Enter your query: ")
            if query.strip().lower() in ("exit", "bye"):
                break

            # Get best result from internet
            top_chunks = self.get_best_result(query)

            reference_text = "\n\n".join(
                f"Source {i+1} ({chunk['source']}):\n{chunk['text']}"
                for i, chunk in enumerate(top_chunks)
            )

            # Feed the best chunked result to LLM to generate a response
            prompt = f"""
                You are an AI assistant.

                User question:
                {query}
                
                Use ONLY the information from the reference text below.
                Be detailed, accurate, and structured.
                Do NOT add information that is not present.
                
                Reference text:
                {reference_text}
                
                Provide a clear and comprehensive answer.
            """
            response = self.gemini_model.models.generate_content(contents=prompt, model=self.gemini_model_name)
            print(response.text)


if __name__ == "__main__":
    queryObject = WebQuerySearch()
    queryObject.main()
