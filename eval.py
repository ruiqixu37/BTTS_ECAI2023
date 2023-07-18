import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def chatgpt_classifier(exemplar_dict, target_text):
    """
    This function takes in a dictionary of exemplars and returns 
    the text style classification of the target text.

    Parameters
    ----------
    exemplar_dict : dictionary
        dictionary of exemplars with the key being the style and 
        the value being a list of exemplars for that style
    target_text : string

    Returns
    -------
    string
        The style of the target text
    """

    # Get the styles from the dictionary
    styles = list(exemplar_dict.keys())

    # Create the prompt in multiline string format
    style_list = ' '.join(styles[:-1]) + ' and ' + styles[-1]
    exemplar_prompt = ''
    for style in styles:
        # Get the list of exemplars for the style
        # and concatenate them into a string
        exemplar_list = exemplar_dict[style]
        if type(exemplar_list) == str:
            exemplar_prompt += f"""'{style}' could be: {exemplar_list}
            """
        else:
            exemplar_prompt += f"""'{style}' could be:
            """
            for exemplar in exemplar_list[:-1]:
                exemplar_prompt += f"""'{exemplar}',
                """
            exemplar_prompt += f"""or '{exemplar_list[-1]}'
            """

    prompt = f"""You are a sentiment classifier.
    Classify the sentiment of the following text into a single word.
    The possible options are '{style_list}'.
    Here are some examples for each sentiment:
    {exemplar_prompt}
    """

    # Create the completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""{prompt}"""},
            {"role": "user", "content": f'{target_text}'}
        ],
        temperature=0.,
    )

    # Return the classification
    style = completion.choices[0].message.content

    if style not in styles:
        return None
    else:
        return style


if __name__ == "__main__":
    # test the classifier with formal and informal exemplars
    exemplar_dict = {
        'formal': ['I am writing to inform you that the package has been delivered.',
                   'Please be advised that your order has been shipped.'],
        'informal': ['Hey, just wanted to let you know that your package has arrived.',
                     'Your order has been shipped!']
    }
    target_text = 'Sir, your package has arrived.'
    print(chatgpt_classifier(exemplar_dict, target_text))

    # test the classifier with positive and negative exemplars
    exemplar_dict = {
        'positive': ['I love you so much!',
                     'You are the best person ever!'],
        'negative': ['I hate you so much!',
                     'You are the worst person ever!']
    }
    target_text = 'I like this movie. 10/10 would watch again.'

    print(chatgpt_classifier(exemplar_dict, target_text))
