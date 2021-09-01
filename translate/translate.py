from transformers import FSMTTokenizer, FSMTForConditionalGeneration

class translator():
    def __init__(self, name_model):
        self.tokenizer = FSMTTokenizer.from_pretrained(name_model)
        self.model = FSMTForConditionalGeneration.from_pretrained(name_model)
    def translate_question(self, question):
        input_ids = self.tokenizer.encode(question, return_tensors="pt")
        outputs = self.model.generate(input_ids)
        decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decode

def translate_batch(batch, ru_eng_translator, eng_ru_translator, ques_or_ans_key = "question"):
    for dict_ in batch:
        if dict_['lang'] == 'ru':
            if ques_or_ans_key == "question":
                dict_[f"{ques_or_ans_key}_translate"] = eng_ru_translator.translate_question(dict_[ques_or_ans_key])
            elif ques_or_ans_key == "answer":
                dict_[f"{ques_or_ans_key}_translate"] = ru_eng_translator.translate_question(dict_[ques_or_ans_key])
        elif dict_['lang'] == 'en':
            if ques_or_ans_key == "question":
                dict_[f"{ques_or_ans_key}_translate"] = ru_eng_translator.translate_question(dict_[ques_or_ans_key])
            elif ques_or_ans_key == "answer":
                dict_[f"{ques_or_ans_key}_translate"] = eng_ru_translator.translate_question(dict_[ques_or_ans_key])

