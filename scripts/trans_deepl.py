# %%
import requests
from dotenv import load_dotenv
import os
from load_file import load_file
import json 
load_dotenv()
X_RAPIDAPI_KEY = os.getenv("X_RAPIDAPI_KEY")
print(X_RAPIDAPI_KEY)
url = "https://deepl-translator.p.rapidapi.com/translate"
SET_SEPERATOR = "{}"
AIHUMAN_SEPERATOR = "인공_지능_휴먼"
# %%
file_name = "chatdoctor200k"
temp_file_name = "chatdoctor200k_temp"
data = load_file(f"../data/chat_doctor/{file_name}.json")
# %%
# 키 값들을 입력으로 받아서, 해당 키값의 value를 {}에 넣어서 String을 이어서 반환하는 함수
def make_payload(sample, keys):
	aihuman_separator = AIHUMAN_SEPERATOR
	payload = ""
	for key in keys:
		payload += f"{sample[key]}{aihuman_separator}"
	payload = payload[:-len(aihuman_separator)]
	return payload, len(payload)

# 여러 대화 samples을 받아 charactor count가 5000이 넘어가지 않게 나눠서 반환하는 함수
def make_payloads(samples, keys):
	set_separator = SET_SEPERATOR
	payloads = []
	nchars = 0
	payload=""
	for sample in samples:
		text, nchar = make_payload(sample, keys)		
		if nchars + nchar + len(set_separator) > 5000:
			payload = payload[:-len(set_separator)]
			payloads.append(payload)
			payload = text
			nchars = nchar
		else:
			payload += f"{text}{set_separator}"
			nchars += nchar + len(set_separator)
	return payloads


# singleset payload
def make_singleset_payloads(samples, keys):
	set_separator = SET_SEPERATOR
	payloads = []
	for sample in samples:
		text, nchar = make_payload(sample, keys)		
		payloads.append(text)
	return payloads

chat_sets = make_singleset_payloads(data,['input','output'])
# %%

headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": X_RAPIDAPI_KEY,
	"X-RapidAPI-Host": "deepl-translator.p.rapidapi.com"
}
payload = {
	"text": "This is a example text for translation.",
	"source": "EN",
	"target": "KO"
}
payloads=[]
for chat_set in chat_sets:
	payloads.append({
		"text": chat_set,
		"source": "EN",
		"target": "KO"
	})
# %%
save_path = f"../data/ko_doctor/{file_name}_ko.jsonl"
temp_save_path = f"../data/ko_doctor/{temp_file_name}_ko.jsonl"


# 기존에 저장된 파일이 있으면, 해당 파일을 불러와서, 마지막 샘플의 번역이 끝난 후부터 다시 번역을 시작한다.
trans_continue = False
if os.path.isfile(temp_save_path):
	trans_continue = True
	with open(temp_save_path, "r") as temp_file:
		temp_lines = temp_file.readlines()
		last2_line = temp_lines[-2]
		last_sample = json.loads(last2_line)
		history_enum = last_sample['enum']

    
# %%
if trans_continue == True:
	mode = 'a'
	print(f"continue from {history_enum}")
else:
    mode = 'w'
with open(save_path, mode) as file, open(temp_save_path, mode) as temp_file:
	prompt_sets = []
	for enum, payload in enumerate(payloads):
		if enum < history_enum:
			continue
		print(enum, payload)
		response = requests.post(url, json=payload, headers=headers)
		# json으로 변환할 수 없는 경우, 에러 메시지를 출력하고 다음 샘플로 넘어간다.
		try:
			response.json()
		except:
			print('error', response)
			continue
		# 'text' key가 없으면, 에러 메시지를 출력하고 다음 샘플로 넘어간다.
		if 'text' not in response.json():
			print('error', response.json())
			continue
		chat_set = response.json()['text']
		one_sample = chat_set.split(AIHUMAN_SEPERATOR)
		if len(one_sample) != 2:
			print('not pair', one_sample)
			continue
		prompt_set = {
			"instruction": "",
			"input": "",
			"output": "",
		}
		prompt_set["instruction"]= one_sample[0]
		prompt_set["output"] = one_sample[1]
		prompt_sets.append(prompt_set)
		json.dump(prompt_set, file,  ensure_ascii=False)
		file.write("\n")
		history = {'enum': enum, 'prompt_set': payload['text']}
		json.dump(history, temp_file,  ensure_ascii=False)
		temp_file.write("\n")
