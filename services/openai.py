from typing import List
import openai


from tenacity import retry, wait_random_exponential, stop_after_attempt

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_SIZE = 1536

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    embeddings = []
    for text in texts:
        if text.strip():
            # Call the OpenAI API to get the embedding for the text
            response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
            # Extract the embedding data from the response
            data = response["data"]  # type: ignore
            # Add the embedding to the embeddings list
            embeddings.append(data[0]["embedding"])
        else:
            # Use a default embedding for empty text
            embeddings.append([0.0] * EMBEDDING_SIZE)
    return embeddings


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    print(f"Completion: {completion}")
    return completion
