import os
import pandas as pd
import time
from langchain.schema import Document
from qdrant_client import QdrantClient
from src.utils.sentence_transformer_instance import get_sentence_transformer
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchText
from src.utils.generic_Utils import CONFIG

COLLECTION_NAME = CONFIG["QDRANT_COLLECTION_NAME"]
_qdrant_client = None
def get_qdrant_client():
    start = time.perf_counter()
    global _qdrant_client
    if _qdrant_client is None:
        QDRANT_CLUSTER_URL =CONFIG["QDRANT_CLUSTER_URL"]
        QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY, timeout=10)
        end = time.perf_counter()
        print(f"Qdrant client initialization completed in {end - start:.2f} seconds.")
    return _qdrant_client

def init_qdrant_indexes():
    """
    This should only be called ONCE inside the lifespan, 
    not at the top of the file!
    """
    start = time.perf_counter()
    client = get_qdrant_client()

    
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="Category",
        field_type="text"
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="CIForNonCIF",
        field_type="keyword"
    )
    end = time.perf_counter()
    print(f"Qdrant indexes initialization completed in {end - start:.2f} seconds.")

def get_search_results(category, cif_type, top_k=5):
    
    # --- Step 2: Define a Query and Embed it ---
    model = get_sentence_transformer()
    query_vector = model.encode(category).tolist()

    # --- Step 3: Perform the Semantic Search with a Filter ---
    print(f"Searching for points similar to: '{category}' with type: '{cif_type}'")

    # Define the filter for 'CIForNonCIF'
    # The 'Both' case requires a special condition to match either 'CIF' or 'Non-CIF'
    if cif_type == "DORA CIF":
        cif_filter = FieldCondition(
            key="CIForNonCIF",
            match=MatchAny(any=["CIF", "Both"])
        )
    else:
        cif_filter = FieldCondition(
            key="CIForNonCIF",
            match=MatchAny(any=["Non CIF", "Both"])
        )
    client = get_qdrant_client()
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="Category",
                    match=MatchText(text=category)
                ),
                cif_filter  # Add the new filter here
            ]
        ),
        limit=top_k
    ).points

    return search_result


def get_enhanced_playbook_category(category="All", cif_type = "Both", top_k=5):
    try:
        search_result = get_search_results(category=category, cif_type=cif_type, top_k=top_k)

        playbook_sections = []
        for point in search_result:
            payload = point.payload

            category = payload.get('Category', '')
            categoryDescription = payload.get('CategoryDescription', '')
            articleReference = payload.get('ArticleReference', '')
            definition = payload.get('ArticleDefinitions', '')
            preferredPositions = payload.get('PreferredPositions', '')

            if category:
                section = f"| Category Name : " + f" {category} | "
                section += f"| Category Description : " + f" {categoryDescription} | "

                if articleReference:
                    section += f"| Article Reference : " + f" {articleReference} | "
                    section += f"| Article Definition : " + f" {definition} | "

                if preferredPositions:
                    section += f"**Preferred Positions:** {preferredPositions}"

                section += " |"
                playbook_sections.append(section)

        return "\n".join(playbook_sections)

    except Exception as e:
        print(f"Error occurred while fetching enhanced playbook: {e}")
        return ""

def get_article_clause_guidance_category(category="All", cif_type="Both", top_k=5):
    try:
        search_result = get_search_results(category=category, cif_type=cif_type, top_k=top_k)

        playbook_sections = []
        for point in search_result:
            payload = point.payload

            category = payload.get('Category', '')
            categoryDescription = payload.get('CategoryDescription', '')
            articleReference = payload.get('ArticleReference', '')
            definition = payload.get('ArticleDefinitions', '')
            preferredPositions = payload.get('PreferredPositions', '')
            clauseGuidanceNode = payload.get('ClauseGuidanceNode', '')
            backStopClauseWording = payload.get('BackStopClauseWording', '')
            backStopClauseLibraryGuidanceNote = payload.get('BackStopClauseLibraryGuidanceNote', '')

            if category:
                section = f"| Category : " + f" {category} | "
                section += f"| Category Description : " + f" {categoryDescription} | "

                if articleReference:
                    section += f"| Article Reference : " + f" {articleReference} | "
                    section += f"| Article Definition : " + f" {definition} | "

                if preferredPositions:
                    section += f"**Preferred Positions:** {preferredPositions}"

                if clauseGuidanceNode:
                    section += f" **Clause Guidance Node:** {clauseGuidanceNode}"

                if backStopClauseWording:
                    section += f" **Back Stop Clause Wording:** {backStopClauseWording}"

                if backStopClauseLibraryGuidanceNote:
                    section += f" **Back Stop Clause Library Guidance Note:** {backStopClauseLibraryGuidanceNote}"

                section += " |"
                playbook_sections.append(section)

        return "\n".join(playbook_sections)

    except Exception as e:
        print(f"Error occurred while fetching enhanced playbook: {e}")
        return ""


def get_enhanced_playbook(query_text="All", top_k=5):
    try:
        search_result = get_search_results(query_text, top_k)

        playbook_sections = []
        for point in search_result:
            payload = point.payload

            article = payload.get('DORAArticle', '')
            definition = payload.get('ArticleExplanation', '')
            preferred = payload.get('PreferredClauseLanguageGuidance', '')
            guidance = payload.get('BackStopClauseLanguageGuidance', '')

            if article:
                section = f"| {article} | "

                if definition:
                    section += f"{definition} | "
                else:
                    section += "N/A | "

                if preferred:
                    section += f"**Preferred:** {preferred}"

                if guidance:
                    section += f" **Guidance:** {guidance}"

                section += " |"
                playbook_sections.append(section)

        return "\n".join(playbook_sections)

    except Exception as e:
        print(f"Error occurred while fetching enhanced playbook: {e}")
        return ""

def get_article_clause_guidance(query_text="All", top_k=5):
    try:
        search_result = get_search_results(query_text, top_k)

        playbook_sections = []
        for point in search_result:
            payload = point.payload

            article = payload.get('DORAArticle', '')
            guidance = payload.get('BackStopClauseLanguageGuidance', '')

            if article:
                section = f"| {article} | "

                if guidance:
                    section += f" **Guidance:** {guidance}"

                section += " |"
                playbook_sections.append(section)

        return "\n".join(playbook_sections)

    except Exception as e:
        print(f"Error occurred while fetching article clause guidance: {e}")
        return ""

def get_unique_categories_by_cif(cif_type="Both"):
    """
    Fetches all unique 'Category' values from the Qdrant collection,
    optionally filtered by the 'CIForNonCIF' field (DORA classification).

    Args:
        cif_type (str): The classification to filter by.
                        - "DORA CIF": Gets categories marked "CIF" or "Both".
                        - "DORA Non-CIF" (or any other string): Gets categories marked "Non CIF" or "Both".
                        - "Both" (or None): Gets all unique categories regardless of CIF type.

    Returns:
        list: A sorted list of unique category names, or an empty list on error.
    """
    print(f"Fetching unique categories for CIF type: '{cif_type}'...")

    # 1. Define the filter based on cif_type, matching your get_search_results logic
    scroll_filter = None
    if cif_type == "DORA CIF":
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="CIForNonCIF",
                    match=MatchAny(any=["CIF", "Both"])
                )
            ]
        )
    elif cif_type == "Both" or cif_type is None:
        scroll_filter = None  # No filter, get all categories
    else:
        # Handles "DORA Non-CIF" or any other value
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="CIForNonCIF",
                    match=MatchAny(any=["Non CIF", "Both"])
                )
            ]
        )

    # 2. Use the scroll API to iterate through all matching points
    unique_categories = set()
    next_offset = None
    client = get_qdrant_client()

    try:
        while True:
            # Request a page of results
            scroll_result = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,  # Process 100 points at a time
                with_payload=["Category"],  # Only fetch the 'Category' field
                with_vectors=False,         # No need for vectors, save bandwidth
                offset=next_offset          # Start from the last page's offset
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            # Add categories from this page to the set
            if not points:
                break  # No more points to fetch

            for point in points:
                category = point.payload.get('Category')
                if category:
                    unique_categories.add(category)

            # If next_offset is None, we've reached the end
            if next_offset is None:
                break

        # Return the unique categories as a sorted list
        return sorted(list(unique_categories))

    except Exception as e:
        print(f"Error fetching unique categories:{e}")
        return []