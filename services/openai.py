from typing import List
import openai


from tenacity import retry, wait_random_exponential, stop_after_attempt

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_SIZE = 1536

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        embeddings = []
        for text in texts:
            if text.strip():
                response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
                if 'data' in response:
                    embeddings.append(response['data'][0]['embedding'])
                else:
                    logger.error(f"OpenAI API response does not contain 'data': {response}")
                    embeddings.append([0.0] * EMBEDDING_SIZE)
            else:
                embeddings.append([0.0] * EMBEDDING_SIZE)
        return embeddings
    except Exception as e:
        logger.error("Failed to get embeddings: {e}", exc_info=True)
        raise e

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
