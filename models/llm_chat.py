import dashscope
from dashscope import Generation
from http import HTTPStatus
import json
import time
import openai
from utils import cloudgpt_aoai_new
from utils.cloudgpt_aoai_new import get_chat_completion

# temperature = 0  # 0 0.7
# top_p = 1  # 0 0.95
# frequency_penalty = 0
# presence_penalty = 0
# MAX_TOKENS=500
#
# limlt_try = 10

dashscope.api_key = os.environ['QWEN_KEY']
# dashscope.api_key = 'sk-ad4c9b3a83ce4d2c9117764d8196a67d'

def get_qwen(prompt, config,call_num=1):
    limlt_try = config["LLM"]["max_retries"]
    llm_parameters = {
        "temperature": config["LLM"]["temperature"],
        "top_p": config["LLM"]["top_p"],
        "top_k": None,

    }

    response = Generation.call(
        model='qwen-plus',  # qwen-plus
        prompt=prompt,
        temperature=llm_parameters["temperature"],
        top_p=llm_parameters["top_p"],
        top_k=llm_parameters["top_k"],
        # max_tokens=config["LLM"]["max_tokens"]
    )
    if response.status_code == HTTPStatus.OK:
        llm_response = json.dumps(response.output, indent=4, ensure_ascii=False)
        res_in_dict = json.loads(llm_response)
        return res_in_dict['text']
    else:
        if call_num >limlt_try:
            print('qwen api time out')
            return ''
        return get_qwen(prompt,config, call_num+1)

def PX_get_chat_completion():
    def test_call(*args, **kwargs):
        test_message = "What is the content?"
        test_chat_message = [{"role": "user", "content": test_message}]

        response = get_chat_completion(
            # model="gpt-4o-mini-20240718",
            model="gpt-35-turbo-20220309",
            # engine="gpt-4-20230613",
            messages=test_chat_message,
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            *args,
            **kwargs,
        )

        print(response.choices[0].message.content)

    print("test without AAD app")
    test_call()  # test without AAD app



def get_chat_completion_with_retry(engine,messages, config,sleep_duration=5):
    retry_count = 0
    while retry_count < config["LLM"]["max_retries"]:
        try:
            response = cloudgpt_aoai_new.get_chat_completion(
                model=engine,
                messages=messages,
                temperature=config["LLM"]["temperature"],  # 0 0.7
                max_tokens=config["LLM"]["max_tokens"],
                top_p=config["LLM"]["top_p"],  # 0 0.95
                frequency_penalty=config["LLM"]["frequency_penalty"],
                presence_penalty=config["LLM"]["presence_penalty"],
                stop=None
            )

            return response.choices[0].message.content
        except openai.RateLimitError:
            print(f"达到速率限制，等待{sleep_duration}秒后重试...")
            time.sleep(sleep_duration)
            retry_count += 1




if __name__ == "__main__":
    config = {
        "LLM": {
            # "name": "qwen",  # gpt4 / gpt3.5
            "engine": "gpt-4-20230613",  # ["gpt-35-turbo-20230613", "gpt-4-20230613"]
            "max_tokens": 1500,
            "prompt_strategy": "guidelines",  # [CoT]
            "temperature": 1,
            "top_p": 1e-10,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_retries": 10
        }
    }

    prompt = 'hello'
    test_chat_message = [{"role": "user", "content": prompt},
                         {"role": "system", "content": prompt}]

    engine = "gpt-4-20230613"
    print(get_chat_completion_with_retry(engine,test_chat_message,config))
