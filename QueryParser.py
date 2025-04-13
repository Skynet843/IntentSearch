import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Define Structured Schema ===
class QuerySchema(BaseModel):
    query: str = Field(..., description="Rewritten search string with semantic understanding")
    category: Optional[str] = Field(None, description="Inferred product category")
    price_max: Optional[int] = Field(None, description="Max price if mentioned")
    intent: Optional[str] = Field(None, description="User's semantic intent like cheap, premium, etc.")

# === Query Rewriter Class ===
class QueryRewriter:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in .env or pass explicitly.")

        self.llm = GoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=self.api_key,
        )

        self.parser = PydanticOutputParser(pydantic_object=QuerySchema)

        self.prompt = PromptTemplate(
            template="""
You are a query parser for an e-commerce product search engine. Even user provide Any Language as query you make it in english.

Analyze the user's input and extract the following fields in valid JSON:
- query: cleaned search string that best captures the user’s need
- category: product category (if possible)
- price_max: numeric price limit (if mentioned)
- intent: e.g. 'cheap', 'premium', 'eco-friendly', etc

{format_instructions}

User Query: {input}
""",
            input_variables=["input"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=verbose)

    def rewrite(self, user_input: str) -> Dict[str, Any]:
        try:
            logger.info(f"🔍 Rewriting query: {user_input}")
            raw_output = self.chain.run({"input": user_input})
            parsed_output = self.parser.parse(raw_output)
            logger.debug(f"✅ Parsed Output: {parsed_output}")
            return parsed_output.dict()
        except ValidationError as ve:
            logger.error("❌ Validation error in LLM response:", exc_info=True)
            return {"query": user_input, "error": "Invalid schema from model", "details": str(ve)}
        except Exception as e:
            logger.error("❌ Exception during query rewrite:", exc_info=True)
            return {"query": user_input, "error": str(e)}
        
if __name__ == "__main__":
    # Example usage
    rewriter = QueryRewriter()
    user_query = "I need a good yoga mat under ₹500 for daily use at home"
    structured_query = rewriter.rewrite(user_query)
    print("\n🔍 Final Structured Query:")
    print(structured_query)