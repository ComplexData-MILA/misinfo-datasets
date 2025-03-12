TEMPLATE_BINARY_NO_SEARCH_NO_DATE = """\
The following statement is going to be given to an AI system to determine if it's true or false and write an explanation why.
Statement: '{statement}'

The only thing the AI will be given is the statement itself, as written above - no context, visuals, or any other information. Your task is to assess if the AI could possibly give a valid answer. Note that this is not about assessing how likely the AI is to give the right answer, but whether it's even possible to evaluate the veracity of the statement based on the information given. The evaluation might be impossible if there is too much ambiguity or missing context.

For example, here is a non-exhaustive list of information that might make it hard to evaluate the veracity of a statement if missing:
1. Identity of a key person, such as the speaker or someone else referenced ambiguously in the statement.
2: Location, if veracity depends on it but it isn't provided. 
3. Textual information or evidence that's mentioned in the statement but not supplied.
4. Visual or audio evidence mentioned in the statement (note that the AI will only be given the statement text). 
5. Temporal information. Note that the date the statement was made is unknown. This might not be relevant, though, if the statement could be evaluated as true or false regardless of when it was made.
6. There's no claim for which evaluating the veracity even makes sense.

Rate on the following scale how possible it seems to evaluate the veracity of the statement:

1: Feasible: There might be some room for interpretation or contextual influence, but the statement is still fairly clear and unambiguous. For example, "the Earth is round" - technically, it is not perfectly round, but in the majority of contexts it would be reasonable to classify this as true.
0: Impossible to evaluate, even with access to external knowledge retrieval systems. There are clearly multiple valid ways the statement could be interpreted that would strongly influence the veracity, mandatory and irrecoverable information is missing, or the statement contains no claim or is downright nonsensical.

Give a brief explanation, then write a vertical bar "|", followed by your rating as a number alone.
"""

TEMPLATE_BINARY_WITH_SEARCH_NO_DATE = """\
The following statement is going to be given to an AI system to determine if it's true or false and write an explanation why.
Statement: '{statement}'

The only thing the AI will be given is the statement itself, as written above - no context, visuals, or any other information. Your task is to assess if the AI could possibly give a valid answer. Note that this is not about assessing how likely the AI is to give the right answer, but whether it's even possible to evaluate the veracity of the statement based on the information given. The AI will have access to a web search system to look for both primary and secondary sources, but the evaluation might still be impossible if there is too much ambiguity or missing context.

For example, here is a non-exhaustive list of information that might make it hard to evaluate the veracity of a statement if missing:
1. Identity of a key person, such as the speaker or someone else referenced ambiguously in the statement.
2: Location, if veracity depends on it but it isn't provided. 
3. Textual information or evidence that's mentioned in the statement but not supplied.
4. Visual or audio evidence mentioned in the statement (note that the AI will only be given the statement text). 
5. Temporal information. Note that the date the statement was made is unknown. This might not be relevant, though, if the statement could be evaluated as true or false regardless of when it was made.
6. There's no claim for which evaluating the veracity even makes sense.

Rate on the following scale how possible it seems to evaluate the veracity of the statement:

1: Feasible, assuming that the retrieval of external knowledge is possible- There is some clear ambiguity, missing context, or multiple potential interpretations. But there seems to be around one-half chance of evaluating the meaning as intended or figuring out the context from a strong knowledge base or web search.
0: Impossible to evaluate, even with access to external knowledge retrieval systems. There are clearly multiple valid ways the statement could be interpreted that would strongly influence the veracity, mandatory and irrecoverable information is missing, or the statement contains no claim or is downright nonsensical.

Give a brief explanation, then write a vertical bar "|", followed by your rating as a number alone.
"""
