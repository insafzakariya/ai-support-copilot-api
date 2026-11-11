from langchain_core.documents import Document

# Created Sample Document insted of using a dasbase or external source for demonstration purposes
DOCUMENTS = [
    Document(
        page_content=(
            "If a payment succeeds but the subscription remains inactive, "
            "verify the payment provider status, then retry subscription "
            "provisioning. Do not ask the customer to pay again."
        ),
        metadata={
            "document_id": "billing-001",
            "title": "Payment succeeded but subscription is inactive",
        },
    ),
    Document(
        page_content=(
            "For duplicate charges, confirm whether one charge is pending. "
            "If two completed charges exist, create a refund request and "
            "tell the customer the normal processing time."
        ),
        metadata={
            "document_id": "billing-002",
            "title": "Duplicate payment handling",
        },
    ),
    Document(
        page_content=(
            "Never request a customer's full card number, CVV, password, or "
            "one-time password. Ask only for the transaction ID or the last "
            "four digits when verification is necessary."
        ),
        metadata={
            "document_id": "security-001",
            "title": "Payment data handling policy",
        },
    ),
]


# Created a function to retrieve documents based on a query for demonstration purposes
def retrive_documents(query: str, limit: int = 5) -> list[Document]:
    """
    Retrieve documents based on a query. This is a simple implementation
    that returns all documents containing the query string in their content.
    """
    query_words = set(query.lower().split())
    scored = []

    for document in DOCUMENTS:
        searchable = document.page_content.lower()
        score = sum(1 for word in query_words if len(word) > 3 and word in searchable)
        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for score, document in scored[:limit] if score > 0]
