"""
Prompt engineering — dynamic chain-of-thought templates,
few-shot injection, and query rewriting.

Design principles:
  1. Every template is versioned so A/B testing via LangSmith is trivial.
  2. Chain-of-thought reasoning is always requested before the final answer.
  3. Hallucination guard: model is explicitly told to say "I don't know"
     if evidence is insufficient.
"""
from __future__ import annotations

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document
# ─── Prompt Versions ──────────────────────────────────────────────────────────

class PromptVersion(str, Enum):
    BASELINE = "baseline_v1"
    COT = "cot_v2"          # Chain-of-thought
    FEW_SHOT_COT = "few_shot_cot_v3"  # Few-shot + CoT (production default)


# ─── System Prompts ───────────────────────────────────────────────────────────

_SYSTEM_BASELINE = """You are a helpful assistant. Answer the user's question
using the provided context. If the context does not contain the answer,
say "I don't know."
"""

_SYSTEM_COT = """You are a precise research assistant. Your job is to answer
questions strictly from the provided context.

Rules:
1. THINK step by step — show your reasoning before your final answer.
2. GROUND every claim in the context; cite the source chunk if available.
3. If the context is insufficient, respond: "I don't have enough information
   to answer this confidently."
4. Do NOT invent facts, statistics, or citations not present in the context.
5. Keep your final answer concise and direct.

Format your response as:
<reasoning>
Step-by-step analysis of the context...
</reasoning>
<answer>
Final concise answer here.
</answer>
"""

_FEW_SHOT_EXAMPLES = """
---
Example 1
Context: "The Eiffel Tower was completed in 1889 and stands 330 metres tall."
Question: When was the Eiffel Tower built?
Reasoning: The context states it was completed in 1889.
Answer: The Eiffel Tower was completed in 1889.

---
Example 2
Context: "Annual revenue for FY2023 was $4.2M, up 18% from the prior year."
Question: What was the revenue growth rate?
Reasoning: The context explicitly states 18% growth year-over-year.
Answer: Revenue grew 18% year-over-year in FY2023.

---
Example 3
Context: "The study examined 200 participants aged 25–40."
Question: What was the average salary of participants?
Reasoning: The context does not mention salary information.
Answer: I don't have enough information to answer this confidently.
---
"""

_SYSTEM_FEW_SHOT_COT = f"""{_SYSTEM_COT}

Here are examples of ideal responses:
{_FEW_SHOT_EXAMPLES}
"""

# ─── Context Formatter ────────────────────────────────────────────────────────

def format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a numbered context block.
    Includes source metadata for citation.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        loc = f"{source}, p.{page}" if page else source
        parts.append(f"[{i}] ({loc})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


# ─── Prompt Builder ───────────────────────────────────────────────────────────

_HUMAN_TEMPLATE = """Context:
{context}

---
Question: {question}

Answer:"""


def build_prompt(
    version: PromptVersion = PromptVersion.FEW_SHOT_COT,
) -> ChatPromptTemplate:
    """Return a versioned ChatPromptTemplate."""
    system_map = {
        PromptVersion.BASELINE: _SYSTEM_BASELINE,
        PromptVersion.COT: _SYSTEM_COT,
        PromptVersion.FEW_SHOT_COT: _SYSTEM_FEW_SHOT_COT,
    }
    system_content = system_map[version]
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_content),
            ("human", _HUMAN_TEMPLATE),
        ]
    )


# ─── Query Rewriter ───────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """You are a search query optimizer.
Rewrite the given question to be more precise and retrieval-friendly.
Output ONLY the rewritten query, nothing else."""

_REWRITE_HUMAN = "Original question: {question}\n\nRewritten query:"


def build_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", _REWRITE_SYSTEM), ("human", _REWRITE_HUMAN)]
    )
