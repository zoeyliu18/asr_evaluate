import io, os, argparse, statistics
import parselmouth
import numpy as np
import seaborn as sns
import jiwer
import pandas as pd

### gather audio_info.txt ###
def gather_audio_info(file):

	directory_dict = {}
	duration_dict = {}
	transcript_dict = {}

	if 'hupa' not in file:
		with io.open(file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split('\t')
				if toks[0].startswith('File') is False or toks[0] != 'File':
					directory_dict[toks[0]] = toks[1]
					duration_dict[toks[0]] = toks[2]
					transcript_dict[toks[0]] = toks[3]

	if 'hupa' in file:
		with io.open(file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split('\t')
				if toks[0].startswith('File') is False or toks[0] != 'File':
					new_toks = []
					for w in toks[ : -1]:
						new_toks.append(w)
					new_transcript = []
					for w in toks[-1].split()[:-2]:
						new_transcript.append(w)
					new_transcript = ' '.join(w for w in new_transcript)
					new_toks.append(new_transcript)
					directory_dict[new_toks[0]] = new_toks[1]
					duration_dict[new_toks[0]] = new_toks[2]
					transcript_dict[new_toks[0]] = new_toks[3]

	return directory_dict, duration_dict, transcript_dict

### gather LANG_lm_ino.txt ###
def gather_lm_info(file):

	ppl_dict = {}
	num_word_dict = {}
	word_type_dict = {}

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')
			if toks[0].startswith('File') is False or toks[0] != 'File':
				ppl_dict[toks[0]] = toks[4]
				num_word_dict[toks[0]] = toks[5]
				word_type_dict[toks[0]] = toks[6]

	return ppl_dict, num_word_dict, word_type_dict

### Read corpus.txt used to train language models ###
def read_corpus(corpus):

	data = []
	with open(corpus, 'rb') as f:
		for line in f:
			toks = line.strip().split()
			for w in toks:
				data.append(w)

	return list(set(data))

### Read pronunciation dictionary ###
def read_pronunciation(lexicon):
	data = []
	with open(lexicon) as f:
		for line in f:
			toks = line.strip().split()
			data.append(toks[0])
	return data

### Read all_utt2spk ###
def read_utt2spk(utt2spk):

	data = {}
	with open(utt2spk) as f:
		for line in f:
			toks = line.strip().split()
			data[toks[0]] = toks[1]

	return data

### Calculate OOV rate of each utterance ###
def compute_oov(transcript, lexicon):

	c = 0

	transcript = transcript.split()
	for w in transcript:
		if w not in lexicon:
			c += 1

	return round(c * 100 / len(transcript), 2)

### Start gathering all features of each audio ###
def gather_features(audio_info_file, lm_audio_info_file, lexicon, utt2spk, output):

	directory_dict, duration_dict, transcript_dict = gather_audio_info(audio_info_file)
	ppl_dict, num_word_dict, word_type_dict = gather_lm_info(lm_audio_info_file)
	ave_pitch_dict = {}
	ave_intensity_dict = {}

#	corpus_data = read_corpus(corpus)
	lexicon = read_pronunciation(lexicon)
	utt2spk = read_utt2spk(utt2spk)

	collect_all_features = []

	for audio, directory in directory_dict.items():
		snd = parselmouth.Sound(directory)
		pitch = snd.to_pitch()
		num = len(pitch.xs())
		ave_pitch = 0
		for tok in pitch.selected_array['frequency']:
			ave_pitch += tok
		ave_pitch = ave_pitch / num
		ave_pitch_dict[audio] = ave_pitch

		intensity = snd.to_intensity()
		num = len(intensity.xs())
		ave_intensity = 0
		for tok in intensity.values.T:
			ave_intensity += tok[0]
		ave_intensity = ave_intensity / num
		ave_intensity_dict[audio] = ave_intensity

		transcript = transcript_dict[audio]
		oov_rate = compute_oov(transcript, lexicon)
		speaker = utt2spk[audio]

		collect_all_features.append([audio, directory, transcript, duration_dict[audio], ave_pitch_dict[audio], ave_intensity_dict[audio], ppl_dict[audio], num_word_dict[audio], word_type_dict[audio], oov_rate, speaker])

	header = ['File', 'Path', 'Transcript', 'Duration', 'Pitch', 'Intensity', 'PPL', 'Num_word', 'Word_type', 'OOV', 'Speaker']

	with io.open(output + 'dates_utterance_features.txt', 'w') as f:
		f.write('\t'.join(w for w in header) + '\n')
		for tok in collect_all_features:
			f.write('\t'.join(str(w) for w in tok) + '\n')

	return collect_all_features

### Gather training data features; average metrics ###

def gather_train_features(file, lexicon, temp_all_features):

	train_data_features = {}
	train_transcript_list = []
	train_duration_list = []
	train_pitch_list = []
	train_intensity_list = []
	train_ppl_list = []
	train_num_word_list = []
	train_word_type_list = []

	with io.open(file) as f:
		for line in f:
			toks = line.strip().split()
			audio = toks[0]
			temp_info = temp_all_features[audio]
			train_transcript_list.append(temp_info[1])
			try:
				train_duration_list.append(float(temp_info[2]))
			except:
				print(len(temp_all_features[audio]))
				print(temp_info)
			train_pitch_list.append(float(temp_info[3]))
			train_intensity_list.append(float(temp_info[4]))
			train_ppl_list.append(float(temp_info[5]))
			train_num_word_list.append(int(temp_info[6]))
			train_word_type_list.append(int(temp_info[7]))

	### Compute OOV rate of the whole training set ###
	all_train_transcript = ' '.join(transcript for transcript in train_transcript_list)
	train_oov = compute_oov(all_train_transcript, lexicon)

	train_ave_duration = statistics.mean(train_duration_list)
	train_ave_pitch = statistics.mean(train_pitch_list)
	train_ave_intensity = statistics.mean(train_intensity_list)
	train_ave_ppl = statistics.mean(train_ppl_list)
	train_ave_num_word = statistics.mean(train_num_word_list)
	train_ave_word_type = statistics.mean(train_word_type_list)

	train_duration_total = sum(train_duration_list)
	train_num_word_total = sum(train_num_word_list)
	train_word_type_total = sum(train_word_type_list)

	return train_ave_duration, train_ave_pitch, train_ave_intensity, train_ave_ppl, train_ave_num_word, train_ave_word_type, train_oov, train_duration_total, train_num_word_total, train_word_type_total

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'ASR model output')

	args = parser.parse_args()

	lang_dir = 'hupa'

	for quality in ['top_tier', 'second_tier']:
		
		header = ['File', 'Path', 'Transcript', 'Duration', 'Pitch', 'Intensity', 'PPL', 'Num_word', 'Word_type', 'OOV', 'Speaker', 'Train_Duration', 'Train_Pitch', 'Train_Intensity', 'Train_PPL', 'Train_Num_word', 'Train_Word_type', 'Train_OOV', 'Train_Duration_Total', 'Train_Num_word_Total', 'Train_Word_type_Total', 'Duration_ratio', 'Pitch_ratio', 'Intensity_ratio', 'PPL_ratio', 'Num_word_ratio', 'Word_type_ratio', 'OOV_ratio', 'WER', 'MER', 'WIL', 'Output', 'Evaluation', 'Lang']
		regression_file = open('data/hupa/hupa_' + quality + '_dates_regression.txt', 'w')
		regression_file.write('\t'.join(w for w in header) + '\n')

		if 'dates_utterance_features.txt' not in os.listdir('data/hupa/'):
			print('generating dates_utterance_features.txt')
			temp_features = gather_features('data/hupa/dates_audio_info.txt', 'data/hupa/hupa_' + 'dates_lm_info.txt', 'data/hupa/local/dict/lexicon.txt', 'data/hupa/all_utt2spk', 'data/hupa/')

		lexicon = read_pronunciation('data/hupa/local/dict/lexicon.txt')

		data = pd.read_csv('data/hupa/dates_utterance_features.txt', sep = '\t')
		audio_list = data['File'].tolist()
		directory_list = data['Path'].tolist()
		transcript_list = data['Transcript'].tolist()
		duration_list = data['Duration'].tolist()
		pitch_list = data['Pitch'].tolist()
		intensity_list = data['Intensity'].tolist()
		ppl_list = data['PPL'].tolist()
		num_word_list = data['Num_word'].tolist()
		word_type_list = data['Word_type'].tolist()
		oov_list = data['OOV'].tolist()
		speaker_list = data['Speaker'].tolist()

		all_features = {}
		for i in range(len(audio_list)):
			all_features[audio_list[i]] = [directory_list[i], transcript_list[i], duration_list[i], pitch_list[i], intensity_list[i], ppl_list[i], num_word_list[i], word_type_list[i], oov_list[i], speaker_list[i]]

		for evaluate_dir in os.listdir('exp/hupa/' + quality + '/'):
			if evaluate_dir == 'dates':
				for output_dir in os.listdir('exp/hupa/' + quality + '/' + evaluate_dir + '/'):
					all_features = {}
					for i in range(len(audio_list)):
						all_features[audio_list[i]] = [directory_list[i], transcript_list[i], duration_list[i], pitch_list[i], intensity_list[i], ppl_list[i], num_word_list[i], word_type_list[i], oov_list[i], speaker_list[i]]

					index = output_dir[6 : ]
					alternative = all_features
					train_ave_duration, train_ave_pitch, train_ave_intensity, train_ave_ppl, train_ave_num_word, train_ave_word_type, train_oov, train_duration_total, train_num_word_total, train_word_type_total = gather_train_features('data/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/train' + index + '/text', lexicon, all_features)
							
					for k, v in all_features.items():
						if len(v) != 10:
							print(k, v)

					best_d = args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/'					
	
					### Double checking if generating predictions for the right file
					number = output_dir[6 : ]
					gold_file = 'data/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/dev' + number + '/text'
					gold_dev = []
					with io.open(gold_file) as f:
						for line in f:
							toks = line.strip().split()
							gold_dev.append(toks[0])

					with io.open(best_d + '/log/decode.1.log', encoding = 'utf-8') as f:
						for line in f:
							utterance = line.strip().split()
							try:
								if utterance[0] in all_features:
									if utterance[0] not in gold_dev:
										print(best_d)
										print(utterance[0])
							except:
								print(utterance)

					with io.open(best_d + '/log/decode.1.log', encoding = 'utf-8') as f:
						for line in f:
							utterance = line.strip().split()
							try:
								if utterance[0] in all_features:
									all_features = {}
									for i in range(len(audio_list)):
										all_features[audio_list[i]] = [directory_list[i], transcript_list[i], duration_list[i], pitch_list[i], intensity_list[i], ppl_list[i], num_word_list[i], word_type_list[i], oov_list[i], speaker_list[i]]

									ground_truth = all_features[utterance[0]][1]
									hypothesis = ' '.join(w for w in utterance[1 : ])
									measures = jiwer.compute_measures(ground_truth, hypothesis)
									wer = measures['wer']
									mer = measures['mer']
									wil = measures['wil']

									info = all_features[utterance[0]]

									info.append(train_ave_duration)
									info.append(train_ave_pitch)
									info.append(train_ave_intensity)
									info.append(train_ave_ppl)
									info.append(train_ave_num_word)
									info.append(train_ave_word_type)
									info.append(train_oov)
									info.append(train_duration_total)
									info.append(train_num_word_total)
									info.append(train_word_type_total)

									info.append(float(info[2]) / train_ave_duration)
									info.append(float(info[3]) / train_ave_pitch)
									info.append(float(info[4]) / train_ave_intensity)
									info.append(float(info[5]) / train_ave_ppl)
									info.append(int(info[6]) / train_ave_num_word)
									info.append(int(info[7]) / train_ave_word_type)
									info.append(float(info[8]) / train_oov)
									if len(info) != 27:
										print(utterance[0], all_features[utterance[0]])
									info.append(wer)
									info.append(mer)
									info.append(wil)
									info.append(output_dir)
									info.append(evaluate_dir)
									info.append(lang_dir)
									info.insert(0, utterance[0])

									regression_file.write('\t'.join(str(w) for w in info) + '\n')

							except:
								pass
