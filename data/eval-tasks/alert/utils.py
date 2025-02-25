import re
def doc_to_text(doc) -> str:
    """
    Remove the chat template to apply the model specific template

    """
    text = doc['prompt']

    pattern = r"### Instruction:\n(.*?)\n### Response:\n"
    
    # Search for the pattern and extract the question
    match = re.match(pattern, text, re.DOTALL)
    if match:
        # Return the captured question part
        return match.group(1).strip()
    else:
        # Return None if the pattern does not match
        assert False

