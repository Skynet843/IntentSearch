import db
from ImageMaster import BLIPCaptionGenerator
from Indexer import ProductSearchIndexer
from AudioMaster import AudioSearchPipeline


ProductSearchIndexer = ProductSearchIndexer()
# AudioSearchPipeline = AudioSearchPipeline()
# BLIPCaptionGenerator = BLIPCaptionGenerator()


def convert_doc(doc):
    doc['id'] = str(doc['_id'])
    del doc['_id']
    return doc


from tqdm import tqdm
import re
from typing import Optional


class TextCleaner:
    def __init__(self):
        # You can add custom normalization dictionaries here if needed
        self.spec_patterns = [
            (r"\b(\d+)[ ]?H\b", r"\1 hours"),
            (r"\bIPX[\d]+\b", "water-resistant"),
            (r"\bBT\b", "Bluetooth"),
            (r"\b(\d{2})K\b", r"\1,000"),  # 30K → 30,000
        ]

    def clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove emojis & non-ASCII
        text = re.sub(r"[^\x00-\x7F]+", " ", text)

        # Normalize price mentions (₹1999 → "under 2000")
        text = re.sub(r"₹[\d,]+", " ", text)

        # Apply product spec normalization
        for pattern, repl in self.spec_patterns:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # Remove special symbols
        text = re.sub(r"[\*\•\|\~\+\=\_]", " ", text)

        # Remove extra punctuation
        text = re.sub(r"[.,;:!?]{2,}", ".", text)

        # Replace multiple spaces with one
        text = re.sub(r"\s+", " ", text)

        # Lowercase and trim
        return text.strip().lower()
from tqdm import tqdm

cleaner = TextCleaner()

def create_index():
    items = list(db.collection.find())
    id_text_list = []

    print(f"🔍 Found {len(items)} items in the database.")
    for item in tqdm(items, desc="🔧 Creating index"):
        item = convert_doc(item)

        product_id = item.get('id')
        title = item.get('title', '')
        description = item.get('description', '')
        bullet_points = item.get('bullet_points', '')
        query = item.get('query', '')
        caption = ""
        
        # 🔹 Extract category names from nested ladder structure
        category_names = []
        try:
            if "category" in item and isinstance(item["category"], list):
                for cat in item["category"]:
                    if "ladder" in cat and isinstance(cat["ladder"], list):
                        category_names.extend(c.get("name", "") for c in cat["ladder"])
        except Exception as e:
            print(f"⚠️ Category parsing failed for {product_id}: {e}")

        # Join all category names into one string
        category_text = " ".join(category_names)

        # 🔹 Combine all parts into a single string
        raw_text = f"{title}. {bullet_points}. {description}. {caption}. {query}. {category_text}"

        # 🔹 Clean the final text before embedding
        cleaned_text = cleaner.clean_text(raw_text)

        if product_id and cleaned_text.strip():
            id_text_list.append((product_id, cleaned_text))

    return id_text_list

if __name__ == "__main__":
    id_text_list = create_index()
    print(len(id_text_list))
    print(id_text_list[:2])
    ProductSearchIndexer.append(id_text_list)
    print(f"✅ Indexed {len(id_text_list)} products.")