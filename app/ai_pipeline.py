from unittest import result

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from .knowledge_base import retrive_documents

from .schemas import TicketAnalysis
from .config import get_settings

settings = get_settings()

llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.7,
    max_tokens=500,
)

analyze_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a support operations assistant.

            Use only the supplied knowledge-base context for company policy and resolution
            advice. If the context does not contain enough information, say that a human
            agent must investigate. Never invent refunds, timelines, or policies.

            Return a useful, polite draft response. Do not request full card numbers,
            passwords, CVV values, or one-time passwords.

            Every citation must refer to one of the supplied documents.
            """,
        ),
        (
            "human",
            """Analyze this customer ticket.

            Ticket subject:
            {subject}

            Customer message:
            {message}

            Knowledge-base context:
            {context}
            """,
        ),
    ]
)

structured_llm = llm.with_structured_output(TicketAnalysis)
analyysis_chain = analyze_prompt | structured_llm


def format_context(documents) -> str:
    if not documents:
        return "No relevant knowledge-base documents were found."

    blocks = []
    for document in documents:
        blocks.append(
            f"Document ID: {document.metadata['document_id']}\n"
            f"Title: {document.metadata['title']}\n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(blocks)


def analyze_ticket(subject: str, message: str) -> TicketAnalysis:
    """
    Analyze a customer support ticket using the AI model and knowledge base.

    Args:
        subject (str): The subject of the customer ticket.
        message (str): The message content of the customer ticket.

    Returns:
        TicketAnalysis: The structured analysis result from the AI model.
    """
    # Retrieve relevant documents from the knowledge base
    documents = retrive_documents(f"{subject} {message}")
    result = analyysis_chain.invoke(
        {
            "subject": subject,
            "message": message,
            "context": format_context(documents),
        }
    )

    return result
