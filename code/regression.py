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

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')
			if toks[0].startswith('File') is False or toks[0] != 'File':
				directory_dict[toks[0]] = toks[1]
				duration_dict[toks[0]] = toks[2]
				transcript_dict[toks[0]] = toks[3]

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

### Calculate OOV rate of each utterance ###
def compute_oov(transcript, corpus_data):

	c = 0

	transcript = transcript.split()
	for w in transcript:
		if w not in corpus_data:
			c += 1

	return round(c * 100 / len(transcript), 2)

### Start gathering all features of each audio ###
def gather_features(audio_info_file, lm_audio_info_file, corpus, output):

	directory_dict, duration_dict, transcript_dict = gather_audio_info(audio_info_file)
	ppl_dict, num_word_dict, word_type_dict = gather_lm_info(lm_audio_info_file)
	ave_pitch_dict = {}
	ave_intensity_dict = {}

	corpus_data = read_corpus(corpus)

	all_features = []

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
		oov_rate = compute_oov(transcript, corpus_data)

		all_features.append([audio, directory, transcript, duration_dict[audio], ave_pitch_dict[audio], ave_intensity_dict[audio], ppl_dict[audio], num_word_dict[audio], word_type_dict[audio], oov_rate])

	header = ['File', 'Path', 'Transcript', 'Duration', 'Pitch', 'Intensity', 'PPL', 'Num_word', 'Word_type', 'OOV']

	with io.open(output + 'utterance_features.txt', 'w') as f:
		f.write('\t'.join(w for w in header) + '\n')
		for tok in all_features:
			f.write('\t'.join(str(w) for w in tok) + '\n')

	return all_features

### Gather training data features; average metrics ###

def gather_train_features(file, corpus_data):

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
			info = all_features[audio]
			train_transcript_list.append(info[1])
			train_duration_list.append(float(info[2]))
			train_pitch_list.append(float(info[3]))
			train_intensity_list.append(float(info[4]))
			train_ppl_list.append(float(info[5]))
			train_num_word_list.append(int(info[6]))
			train_word_type_list.append(int(info[7]))

	### Compute OOV rate of the whole training set ###
	all_train_transcript = ' '.join(transcript for transcript in train_transcript_list)
	train_oov = float(compute_oov(all_train_transcript, corpus_data))

	train_ave_duration = statistics.mean(train_duration_list)
	train_ave_pitch = statistics.mean(train_pitch_list)
	train_ave_intensity = statistics.mean(train_intensity_list)
	train_ave_ppl = statistics.mean(train_ppl_list)
	train_ave_num_word = statistics.mean(train_num_word_list)
	train_ave_word_type = statistics.mean(train_word_type_list)

	return train_ave_duration, train_ave_pitch, train_ave_intensity, train_ave_ppl, train_ave_num_word, train_ave_word_type, train_oov

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'ASR model output')
	parser.add_argument('--state', type = str, help = 'whethere generating regression data or training regression models')
#	parser.add_argument('--lang', type = str, help = 'language')

	args = parser.parse_args()

	if args.state == 'g':
		for lang_dir in os.listdir(args.input):
			if lang_dir in ['fongbe', 'iban']: #'wolof', 'iban',
				regression_data = []

				if 'utterance_features.txt' not in 'data/' + lang_dir + '/':
					all_features = gather_features('data/' + lang_dir + '/audio_info.txt', 'data/' + lang_dir + '/' + lang_dir + '_lm_info.txt', 'data/' + lang_dir + '/local/corpus.txt', 'data/' + lang_dir + '/')

				corpus_data = read_corpus('data/' + lang_dir + '/local/corpus.txt')

				data = pd.read_csv('data/' + lang_dir + '/utterance_features.txt', sep = '\t')
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

				all_features = {}
				for i in range(len(audio_list)):
					all_features[audio_list[i]] = [directory_list[i], transcript_list[i], duration_list[i], pitch_list[i], intensity_list[i], ppl_list[i], num_word_list[i], word_type_list[i], oov_list[i]]

				for evaluate_dir in os.listdir(args.input + lang_dir + '/'):
					for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):

						index = output_dir[6 : ]

						train_ave_duration, train_ave_pitch, train_ave_intensity, train_ave_ppl, train_ave_num_word, train_ave_word_type, train_oov = gather_train_features('data/' + lang_dir + '/' + evaluate_dir + '/train' + index + '/text', corpus_data)

						wers = []

						with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
							for line in f:
								toks = line.split()
								wers.append(toks[1])

						min_wer = min(wers)
						min_wer_idx = wers.index(min_wer)

						best_d = evaluations[min_wer_idx][-1].split('/')[: -1]
						best_d = '/'.join(w for w in best_d)

						pred_dict = {}

						with io.open(best_d + '/log/decode.1.log', encoding = 'utf-8') as f:
							for line in f:
								try:
									utterance = line.strip().split()
									if utterance[0] in all_features:
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

										info.append(info[3] / train_ave_duration)
										info.append(info[4] / train_ave_pitch)
										info.append(info[5] / train_ave_intensity)
										info.append(info[6] / train_ave_ppl)
										info.append(info[7] / train_ave_num_word)
										info.append(info[8] / train_ave_word_type)
										info.append(info[9] / train_oov)

										info.append(wer)
										info.append(mer)
										info.append(wil)
										info.append(output_dir)
										info.append(evaluate_dir)
										info.append(lang_dir)

										regression_data.append(info)
								except:
									pass

				header = ['File', 'Path', 'Transcript', 'Duration', 'Pitch', 'Intensity', 'PPL', 'Num_word', 'Word_type', 'OOV', 'Train_Duration', 'Train_Pitch', 'Train_Intensity', 'Train_PPL', 'Train_Num_word', 'Train_Word_type', 'Train_OOV', 'Duration_ratio', 'Pitch_ratio', 'Intensity_ratio', 'PPL_ratio', 'Num_word_ratio', 'Word_type_ratio', 'OOV_ratio', 'WER', 'MER', 'WIL', 'Evaluation', 'Lang']

				with io.open('data/' + lang_dir + '/regression.txt', 'w') as f:
					f.write('\t'.join(w for w in header) + '\n')

					for tok in regression_data:
						f.write('\t'.join(str(w for w in tok)) + '\n')
