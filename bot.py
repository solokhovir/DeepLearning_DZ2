import telebot
import onnxruntime
import sentencepiece as spm
import numpy as np
import pickle
import re

flag = 0
#далее указывются названия используемых моделей и служебных файлов, лишнее просто закомментировать
sp = spm.SentencePieceProcessor('m.model') #для генерации текста
sess = onnxruntime.InferenceSession('Conv_next_token.onnx') #для генерации текста(1)
# sess_enc = onnxruntime.InferenceSession('seq2seq_enc.onnx') #для машинного перевода
# sess_dec = onnxruntime.InferenceSession('seq2seq_dec.onnx') #для машинного перевода
sess1 = onnxruntime.InferenceSession('LSTM_next_token.onnx') #для генерации текста(2)
sess2 = onnxruntime.InferenceSession('Transformer_next_token.onnx') #для генерации текста(3)
# with open('vocabs.pickle', 'rb') as f: #для машинного перевода
# 	SRC_SOS, SRC_EOS, SRC_STOI, TRG_SOS, TRG_EOS, TRG_STOI, TRG_ITOS = pickle.load(f)
inputs = sess.get_inputs()
#ниже указать ранее сгенерированный bot_father'ом токен вашего бота
bot = telebot.TeleBot('5268313234:AAG38FsS1hkJ4t8qilte6g69KXdro3sh-fE')

def tokenize(text):
	return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",text)


def preprocess(text):
	tokens = [t.lower() for t in tokenize(text)]
	tokens = [SRC_SOS] + tokens + [SRC_EOS]
	src_indexes = [SRC_STOI.get(token, 0) for token in tokens]
	src_tensor = np.int64(src_indexes).reshape(1, -1)
	src_mask = (np.int64(src_indexes) != 1).reshape(1, 1, 1, -1)
	return src_tensor, src_mask


def get_trg_mask(trg_tensor):
	trg_pad_mask = (trg_tensor != 1).reshape(1, 1, 1, -1)
	trg_len = trg_tensor.shape[1]
	trg_sub_mask = np.tril(np.ones((trg_len, trg_len), dtype=np.bool))
	return trg_pad_mask & trg_sub_mask

def Translate(message):

	test_text = message.text
	print(test_text)
	src_tensor, src_mask = preprocess(test_text)
	enc_src = sess_enc.run(None, {'src_tensor': src_tensor,
                              'src_mask': src_mask})[0]
	trg_indexes = [TRG_STOI[TRG_SOS]]
	for i in range(128):
		trg_tensor = np.int64(trg_indexes).reshape(1, -1)
		trg_mask = get_trg_mask(trg_tensor)
		output, attention = sess_dec.run(None, {'trg_tensor': trg_tensor, 
                                            'enc_src': enc_src,
                                            'trg_mask': trg_mask,
                                            'src_mask': src_mask})
		pred_token = output.argmax(axis=2)[:,-1].item()
		print(pred_token)
		trg_indexes.append(pred_token)
		if pred_token == TRG_STOI[TRG_EOS]:
			break

	trg_tokens = [TRG_ITOS[i] for i in trg_indexes]
	' '.join(trg_tokens[1:-1])
	result = ' '
	for i in range(len(trg_tokens[1:-1])):
		result = result + ' '+ trg_tokens[i+1]
	bot.send_message(message.from_user.id, result)


def Conv1d(message):

	try:
		inputs = sp.encode(message.text)[-16:]
		inputs = [0]*max(16 - len(inputs), 0) + inputs
		finalresult = message.text
		for i in range(10):
			token = sess.run(None, {'input.1': np.array(inputs, dtype=np.int64).reshape(1, 16)})[0]
			inputs.pop(0)
			inputs.append(int(token[-1].argmax()))
			finalresult = finalresult +' '+ sp.decode([int(token[-1].argmax())])
		bot.send_message(message.from_user.id, finalresult)
	except (onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument):
		bot.send_message(message.from_user.id, "Слишком большая последовательность или неверный аргумент")


def LSTM(message):

	inputs = sp.encode(message.text)[-16:]
	inputs = [0]*max(16 - len(inputs), 0) + inputs
	finalresult = message.text 
	for i in range(10):
		token = sess1.run(None, {'input.1': np.array(inputs, dtype=np.int64).reshape(1, 16),
                       'onnx::Slice_1': np.zeros((2, 16, 256), dtype=np.float32),
                       'onnx::Slice_2': np.zeros((2, 16, 256), dtype=np.float32)})[0]
		inputs.pop(0)
		inputs.append(int(token[-1].argmax()))
		print(sp.decode([int(token[-1].argmax())]))
		finalresult = finalresult +' '+ sp.decode([int(token[-1].argmax())])
	bot.send_message(message.from_user.id, finalresult)


def Transformer(message):
	
	inputs = sp.encode(message.text)[-16:]
	inputs = [0]*max(16 - len(inputs), 0) + inputs
	token = sess2.run(None, {'src': np.array(inputs, dtype=np.int64).reshape(1, 16)})[0]
	finalresult = message.text 
	finalresult = finalresult + sp.decode([int(tok.argmax()) for tok in token[0]])
	bot.send_message(message.from_user.id, finalresult)

#данный код предназначен для решения задачи генерации текста
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
	global flag
	if flag == 0:
		if message.text == "/start":
			bot.send_message(message.from_user.id, "Приветствую! Данный бот предназначен для генерации текста. Для справки введи команду: /help")
		elif message.text == "/help":
			bot.send_message(message.from_user.id, "Для перехода в режим генерации текста введи следующую команду: /text_generator")
		elif message.text == "/text_generator":
			bot.send_message(message.from_user.id, "Напиши текст длиною не более 16 слов, который мне нужно продолжить")
			flag = 1
		else:
			bot.send_message(message.from_user.id, "Не понимаю, что тебе нужно. Для справки введи команду: /help")
	elif flag==1:
		#выбрать нужную модель, остальное закомментировать
		#Conv1d(message)
		#LSTM(message)
		Transformer(message)
		flag=0


bot.polling(none_stop=True, interval=0)