import re


def formatting_prompts_func(example, only_function_call_tuning=True):
    USER = 'USER'
    ASSISTANT = 'ASSISTANT'

    output_texts = []
    system, chat = example['system'], example['chat']
    request_search_str = 'USER:(.*)'
    request_answer_str = 'ASSISTANT:(.*)'

    for index, (curr_system, curr_chat) in enumerate(zip(system, chat)):
        user_request = re.findall(request_search_str, curr_chat)
        assistant_response = re.findall(request_answer_str, curr_chat)
        for ur, ar in zip(user_request, assistant_response):
            if only_function_call_tuning and 'functioncall' not in ar:
                continue

            instruction_str = f"{curr_system}{USER}:{ur}"
            response_str = f"{ASSISTANT}:{ar}"
            instruction_str = instruction_str.replace('<|endoftext|>', '</s>')
            response_str = response_str.replace('<|endoftext|>', '</s>')
            prompt = "\n\n".join([i for i in [instruction_str, response_str] if i is not None])
            output_texts.append(prompt)

    return output_texts
