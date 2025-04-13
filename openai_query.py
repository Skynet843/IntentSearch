from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template("""
You are a query rewriter and intent extractor for a product search engine.

Given a user query, return a JSON object with the following fields:
- query: rewritten, cleaned search string
- category: optional inferred category (e.g., 'yoga', 'electronics')
- price_max: optional price cap
- intent: short descriptor like 'affordable', 'premium', 'eco-friendly', etc.

Respond ONLY with valid JSON.

User Query: {input}
""")

import os
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

chain = LLMChain(llm=llm, prompt=prompt)



raw_input = "I need a good yoga mat under ₹500 for daily use at home"

result = chain.run(input=raw_input)
print(result)

import json

def parse_query_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"query": output}  # fallback

structured = parse_query_output(result)