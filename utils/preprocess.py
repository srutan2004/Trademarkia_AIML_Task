import re


def clean_text(text):
    """
    Cleans raw newsgroup posts.

    Cleaning decisions (important for assignment justification):

    1. Headers, quotes, and signatures are removed because they contain
       metadata like email routes, which do not represent semantic content.

    2. URLs and email addresses are removed since they introduce noise
       unrelated to the topic of the message.

    3. Special characters are removed to reduce embedding noise.

    4. Very short documents will be filtered later because they provide
       insufficient semantic context.
    """

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()