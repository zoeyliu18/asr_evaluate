import io, os, argparse, random, librosa, statistics, glob
from pydub import AudioSegment
import parselmouth
import numpy as np
import seaborn as sns

from collections import Counter
from scipy.stats import wasserstein_distance

import collections
from typing import Dict, Generator, Iterator, List, Set, Text, Tuple

import pandas as pd
from scipy import stats
from sklearn import feature_extraction
from sklearn import neighbors

HELDOUT_RATE = 0.2

def gather_audio_info(file):

	data = []

	if 'lm' not in file:
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
					data.append(new_toks)
	else:
		with io.open(file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split('\t')
				if toks[0].startswith('File') is False or toks[0] != 'File':
					new_toks = []
					for w in toks[ : 3]:
						new_toks.append(w)
					new_transcript = []
					for w in toks[3].split()[:-2]:
						new_transcript.append(w)
					new_transcript = ' '.join(w for w in new_transcript)
					new_toks.append(new_transcript)
					for w in toks[-3 : ]:
						new_toks.append(w)
					data.append(new_toks)

	return data

def sort(data, num_of_speakers):

	new_data = []
	text_ids = []

	for idx, text in data.items():
		while idx[0] == '0':
			idx = idx[1 :]
		text_ids.append(int(idx))

	text_ids.sort()

	for i in range(len(text_ids)):
		idx = text_ids[i]
		new_idx = ''		
		if len(str(idx)) == 1:
			new_idx = '000' + str(idx)
		if len(str(idx)) == 2:
			new_idx = '00' + str(idx)
		if len(str(idx)) == 3:
			new_idx = '0' + str(idx)
		if len(str(idx)) == 4:
			new_idx = str(idx)
		
		new_data.append('verdena_' + num_of_speakers + '_' + new_idx + ' ' + ' '.join(w for w in data[str(idx)]))

	return new_data

### Get corpus from training data ###

def text_to_corpus(text):

	corpus = []

	with io.open(text, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			transcripts = toks[1 : -2]
			transcripts = ' '.join(w for w in transcripts)
			corpus.append(transcripts)

	return corpus

def write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output):

	with io.open(output + 'train' + split_id + '/text', 'w', encoding = 'utf-8') as f:
		for tok in train_texts:
			f.write(tok + '\n')

	with io.open(output + 'train' + split_id + '/wav.scp', 'w', encoding = 'utf-8') as f:
		for tok in train_wav:
			f.write(tok + '\n')

	with io.open(output + 'dev' + split_id + '/text', 'w', encoding = 'utf-8') as f:
		for tok in dev_texts:
			f.write(tok + '\n')

	with io.open(output + 'dev' + split_id + '/wav.scp', 'w', encoding = 'utf-8') as f:
		for tok in dev_wav:
			f.write(tok + '\n')

def write_utt_spk(output, split_id, split, speaker, lang):

	with io.open(output + lang + '_' + split + '_' + speaker + '_utt_spk' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
		f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
		f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
		f.write('do' + '\n')
		f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
		f.write("	cat $dir/text | cut -d'_' -f2 > $dir/spk" + '\n')
		f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
		f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
		f.write('	rm $dir/utt $dir/spk' + '\n')
		f.write('done' + '\n')

	sub_dir = output.split('/')[-2] + '/'

	with io.open(output + lang + '_' + split + '_' + speaker + '_compute_mfcc' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
		f.write("echo 'compute mfcc for train dev...'" + '\n')
		f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
		f.write('do' + '\n')
		f.write("	steps/make_mfcc.sh --nj 4 $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("	steps/compute_cmvn_stats.sh $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("done" + '\n')

### Random splits ###

def random_split(file, num_of_speakers, audio_info_data, output, split_id):

	all_audios = []
	all_texts = {}
	all_directories = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	data = []
	time_list = []

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			data.append(toks)
			time_list.append(float(toks[-1]) - float(toks[-2]))

	time = sum(time_list)
	train_time = time * (1 - HELDOUT_RATE)
	dev_time = time - train_time

	train_data = {}
	dev_data = {}

	index_list = []
	for i in range(len(data)):
		index_list.append(i)

	random.shuffle(index_list)

	start = 0
	c = 0

	for i in range(len(index_list)):
		while start <= dev_time:
			tok = data[index_list[i]][0].split('_')
			idx = tok[-1]
			while idx[0] == '0':
				idx = idx[1 :]
			if idx in dev_data:
				print(dev_data[idx])
			dev_data[idx] = data[index_list[i]][1 : -2]
			start += time_list[index_list[i]]
			index_list.remove(index_list[i])
			c += 1

	for i in index_list:
		tok = data[i][0].split('_')
		idx = tok[-1]
		while idx[0] == '0':
			idx = idx[1 :]
		if idx in train_data:
			print(train_data[idx])
		train_data[idx] = data[i][1 : -2]

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

#	return sort(train_data, num_of_speakers), sort(dev_data, num_of_speakers)


### Splitting by threshold of duration ###
### performing speaker normalization separately ###
### for audios of different speakers ###


def duration_threshold_different(audio_info_data, output, split_id, num_of_speakers, lang): 

	duration_list = []

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		duration_list.append(audio_time)

	duration_list.sort()

	dev_size = int(len(duration_list) * HELDOUT_RATE)

	temp_time = 0
	cutoff = 0
	current_count = 0

	i = len(duration_list) - 1

	while temp_time <= int(sum(duration_list) * HELDOUT_RATE):	
		temp_time += duration_list[i]
		i = i - 1

	cutoff = duration_list[i]
	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_durations[audio[0]] = audio_time
		time += audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	for k, v in all_durations.items():
		if v < cutoff:
			train_time += v
			audio = k 
			train_audio.append(audio)

		if v >= cutoff:
			dev_time += v
			audio = k 
			dev_audio.append(audio)

	train_duration = []
	dev_duration = []

	for audio in train_audio:
		train_duration.append(all_durations[audio])

	for audio in dev_audio:
		dev_duration.append(all_durations[audio])

	train_duration.sort()
	dev_duration.sort()
	print(train_duration[-1])
	print(dev_duration[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 


### Splitting by Wasserstein distance, so that the text distributions of train and test are divergent ###

def split_with_wasserstein(audio_info_data, output, split_id, num_of_speakers, lang, no_of_trials = 1, min_df = 1, leaf_size = 3): 

	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	text_data = []

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_durations[audio[0]] = audio_time
		time += audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]
		text_data.append(audio[-1])

	vectorizer = feature_extraction.text.CountVectorizer(dtype=np.int8, min_df=min_df)
	text_counts = vectorizer.fit_transform(text_data)
	text_counts = text_counts.todense()
	heldout_rate = 0.21
	nn_tree = neighbors.NearestNeighbors(n_neighbors=int(len(text_data) * heldout_rate), algorithm='ball_tree', leaf_size=leaf_size, metric=stats.wasserstein_distance)
	nn_tree.fit(text_counts)
	dev_set_indices = []

	for trial in range(no_of_trials):
		# Sample random test centroid.
		sampled_poind = np.random.randint(
			text_counts.max().max() + 1, size=(1, text_counts.shape[1]))
		nearest_neighbors = nn_tree.kneighbors(sampled_poind, return_distance=False)
		# We queried for only one datapoint.
		nearest_neighbors = nearest_neighbors[0]
		dev_set_indices.append(nearest_neighbors)

	train_time = 0
	dev_time = 0
	train_audio = []
	dev_audio = []

	index_list = []
	for i in range(len(text_data)):
		index_list.append(i)

	while dev_time <= time * heldout_rate:
		for idx in index_list:
			if idx in dev_set_indices[0]:
				dev_text = text_data[idx]
				for k, v in all_texts.items():
					if v == dev_text:
						if dev_time <= time * heldout_rate:
							dev_audio.append(k)
							dev_time += all_durations[k]
							if dev_time >= time * heldout_rate:
								break

	for k, v in all_texts.items():
		if k not in dev_audio:
			train_time += all_durations[k]
			train_audio.append(k)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

### Splitting by threshold of utterance perplexity, scored by the language model ###

def ppl_threshold_different(lm_info_data, output, split_id, num_of_speakers, lang):

	ppl_list = []
	ppl_dict = {}
	duration_list = []
	all_durations = {}
	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}

	for i in range(len(lm_info_data)):
		audio = lm_info_data[i]
		audio_ppl = float(audio[-3])
		duration_list.append(float(audio[2]))
		ppl_list.append(audio_ppl)
		ppl_dict[audio[0]] = audio_ppl

		audio_time = float(audio[2])
		time += audio_time
		all_durations[audio[0]] = audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[3]

	ppl_list.sort()

	dev_size = int(len(ppl_list) * HELDOUT_RATE)

	cutoff = ppl_list[-1 * dev_size]

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	i = len(ppl_list) - 1 

	while dev_time <= time * HELDOUT_RATE:
		for k, v in ppl_dict.items():				
			if v == ppl_list[i]:	
				audio = k
				if audio not in dev_audio:	
					dev_time += all_durations[audio]				 
					dev_audio.append(audio)
					if dev_time >= time * HELDOUT_RATE:
						break
		i = i - 1

	for k, v in ppl_dict.items():
		if k not in dev_audio:
			audio = k 
			train_time += all_durations[audio]
			train_audio.append(audio)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_ppl = []
	dev_ppl = []

	for audio in train_audio:
		train_ppl.append(ppl_dict[audio])

	for audio in dev_audio:
		dev_ppl.append(ppl_dict[audio])

	train_ppl.sort()
	dev_ppl.sort()
	print(train_ppl[-1])
	print(dev_ppl[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 


### Splitting by threshold of number of words within utterances###

def num_word_threshold_different(lm_info_data, output, split_id, num_of_speakers, lang):

	num_word_list = []
	num_word_dict = {}
	duration_list = []
	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	for i in range(len(lm_info_data)):
		audio = lm_info_data[i]
		audio_num_word = int(audio[-2])
		duration_list.append(float(audio[2]))
		num_word_list.append(audio_num_word)
		num_word_dict[audio[0]] = audio_num_word

		audio_time = float(audio[2])
		time += audio_time
		all_durations[audio[0]] = audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[3]

	num_word_list.sort()

	dev_size = int(len(num_word_list) * HELDOUT_RATE)

	cutoff = num_word_list[-1 * dev_size]	

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	i = len(num_word_list) - 1 

	while dev_time <= time * HELDOUT_RATE:
		for k, v in num_word_dict.items():				
			if v == num_word_list[i]:		
				audio = k
				if audio not in dev_audio:		
					dev_time += all_durations[audio]				 
					dev_audio.append(audio)
					if dev_time >= time * HELDOUT_RATE:
						break
		i = i - 1

	for k, v in num_word_dict.items():
		if k not in dev_audio:
			audio = k 
			train_time += all_durations[audio]
			train_audio.append(audio)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_num_word = []
	dev_num_word = []

	for audio in train_audio:
		train_num_word.append(num_word_dict[audio])

	for audio in dev_audio:
		dev_num_word.append(num_word_dict[audio])

	train_num_word.sort()
	dev_num_word.sort()
	print(train_num_word[-1])
	print(dev_num_word[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 


### Splitting by threshold of number of word types within utterances###

def word_type_threshold_different(lm_info_data, output, split_id, num_of_speakers, lang):

	word_types_list = []
	word_types_dict = {}
	duration_list = []
	all_durations = {}

	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}

	for i in range(len(lm_info_data)):
		audio = lm_info_data[i]
		audio_word_types = int(audio[-1])
		duration_list.append(float(audio[2]))
		word_types_list.append(audio_word_types)
		word_types_dict[audio[0]] = audio_word_types

		audio_time = float(audio[2])
		time += audio_time
		all_durations[audio[0]] = audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[3]

	word_types_list.sort()

	dev_size = int(len(word_types_list) * HELDOUT_RATE)

	cutoff = word_types_list[-1 * dev_size]

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	i = len(word_types_list) - 1 
	while dev_time <= time * HELDOUT_RATE:
		for k, v in word_types_dict.items():				
			if v == word_types_list[i]:		
				audio = k
				if audio not in dev_audio:	
					dev_time += all_durations[audio]				 
					dev_audio.append(audio)
					if dev_time >= time * HELDOUT_RATE:
						break
		i = i - 1

	for k, v in word_types_dict.items():
		if k not in dev_audio:
			audio = k 
			train_time += all_durations[audio]		
			train_audio.append(audio)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_word_types = []
	dev_word_types = []

	for audio in train_audio:
		train_word_types.append(word_types_dict[audio])

	for audio in dev_audio:
		dev_word_types.append(word_types_dict[audio])

	train_word_types.sort()
	dev_word_types.sort()
	print(train_word_types[-1])
	print(dev_word_types[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

### Split by average intensity ###

def average_intensity_threshold_different(audio_info_data, output, split_id, num_of_speakers, lang):

	ave_intensity_list = []
	ave_intensity_dict = {}
	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		directory = audio[1]
		snd = parselmouth.Sound(directory)
		intensity = snd.to_intensity()
		num = len(intensity.xs())
		ave_intensity = 0
		for tok in intensity.values.T:
			ave_intensity += tok[0]
		ave_intensity = ave_intensity / num 
		ave_intensity_list.append(ave_intensity)
		ave_intensity_dict[audio[0]] = ave_intensity

		audio_time = float(audio[-2])
		time += audio_time
		all_durations[audio[0]] = audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	ave_intensity_list.sort()

	dev_size = int(len(ave_intensity_list) * HELDOUT_RATE)

	cutoff = ave_intensity_list[-1 * dev_size]

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	i = len(ave_intensity_list) - 1 
	while dev_time <= time * HELDOUT_RATE:
		for k, v in ave_intensity_dict.items():
			if v == ave_intensity_list[i]:
				audio = k
				if audio not in dev_audio:
					dev_time += all_durations[audio]
					dev_audio.append(audio)
					if dev_time >= time * HELDOUT_RATE:
						break
		i = i - 1

	for k, v in ave_intensity_dict.items():
		if k not in dev_audio:
			train_time += all_durations[k]
			audio = k 
			train_audio.append(audio)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_intensity = []
	dev_intensity = []

	for audio in train_audio:
		train_intensity.append(ave_intensity_dict[audio])

	for audio in dev_audio:
		dev_intensity.append(ave_intensity_dict[audio])

	train_intensity.sort()
	dev_intensity.sort()
	print(train_intensity[-1])
	print(dev_intensity[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

### Split by average pitch ###

def average_pitch_threshold_different(audio_info_data, output, split_id, num_of_speakers, lang):

	ave_pitch_list = []
	ave_pitch_dict = {}
	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		directory = audio[1]
		snd = parselmouth.Sound(directory)
		pitch = snd.to_pitch()
		num = len(pitch.xs())
		ave_pitch = 0
		for tok in pitch.selected_array['frequency']:
			ave_pitch += tok
		ave_pitch = ave_pitch / num 
		ave_pitch_list.append(ave_pitch)
		ave_pitch_dict[audio[0]] = ave_pitch

		audio_time = float(audio[-2])
		time += audio_time
		all_durations[audio[0]] = audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	ave_pitch_list.sort()

	dev_size = int(len(ave_pitch_list) * HELDOUT_RATE)

	train_time = 0
	dev_time = 0

	train_audio = []
	dev_audio = []

	i = len(ave_pitch_list) - 1 

	while dev_time <= time * HELDOUT_RATE:
		for k, v in ave_pitch_dict.items():				
			if v == ave_pitch_list[i]:	
				audio = k
				if audio not in dev_audio:	
					dev_time += all_durations[audio]				 
					dev_audio.append(audio)
					if dev_time >= time * HELDOUT_RATE:
						break
		i = i - 1

	for k, v in ave_pitch_dict.items():
		if k not in dev_audio:
			train_time += all_durations[k]
			audio = k 
			train_audio.append(audio)

	dev_audio = []
	dev_time = 0
	for k, v in ave_pitch_dict.items():
		if k not in train_audio:
			dev_time += all_durations[k]
			dev_audio.append(k)

	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_pitch = []
	dev_pitch = []

	for audio in train_audio:
		train_pitch.append(ave_pitch_dict[audio])

	for audio in dev_audio:
		dev_pitch.append(ave_pitch_dict[audio])

	train_pitch.sort()
	dev_pitch.sort()
	print(train_pitch[-1])
	print(dev_pitch[0])

	train_data = {}
	dev_data = {}

	for audio in train_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		train_data[idx] = all_texts[audio].split()

	for audio in dev_audio:
		idx = audio.split('_')[-1]
		while idx[0] == '0':
			idx = idx[1 : ]
		dev_data[idx] = all_texts[audio].split()

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

### Get all fieldwork dates ###

def all_dates(path):

	dates = []

	for folder in os.listdir(path):
		if folder.startswith('verdena_') is False and folder.endswith('txt') is False and len(folder) == 6:
			dates.append(folder)

	return dates


### Splits by fieldwork dates ###

def date_split(path, date, num_of_speakers):
	if not os.path.exists('data/hupa/' + quality + '/dates/train' + date + '/'):
		os.makedirs('data/hupa/' + quality + '/dates/train' + date + '/')

	if not os.path.exists('data/hupa/' + quality + '/dates/dev' + date + '/'):
		os.makedirs('data/hupa/' + quality + '/dates/dev' + date + '/')

	if not os.path.exists('data/hupa/' + quality + '/dates/train' + date + '/' + args.n + '/'):
		os.makedirs('data/hupa/' + quality + '/dates/train' + date + '/' + args.n + '/')

	if not os.path.exists('data/hupa/' + quality + '/dates/dev' + date + '/' + args.n + '/'):
		os.makedirs('data/hupa/' + quality + '/dates/dev' + date + '/' + args.n + '/')

	train_data = {}
	dev_data = {}

	for folder in os.listdir(path):

		if folder.startswith('verdena_') is False and folder.endswith('txt') is False and folder != date and len(folder) == 6:
			for file in os.listdir(path + folder):
				if 'text' in file:
					with io.open(path + folder + '/' + file, encoding = 'utf-8') as f:
						for line in f:
							toks = line.strip().split()
							
							index = toks[0].split('_')
							idx = index[-1]
						
							while idx[0] == '0':
								idx = idx[1 :]
							if idx in train_data:
								print(train_data[idx])
						
							train_data[idx] = toks[1 : -2]

				if 'wav' in file:
					os.system('cp ' + path + folder + '/' + file + ' data/hupa/' + quality + '/dates/train' + date + '/' + args.n + '/')

		if folder.startswith('verdena_') is False and folder.endswith('txt') is False and folder == date and len(folder) == 6:
			for file in os.listdir(path + folder):
				if 'text' in file:
					with io.open(path + folder + '/' + file, encoding = 'utf-8') as f:
						for line in f:
							toks = line.strip().split()
							
							index = toks[0].split('_')
							idx = index[-1]
						
							while idx[0] == '0':
								idx = idx[1 :]
							if idx in train_data:
								print(train_data[idx])
						
							dev_data[idx] = toks[1 : -2]

				if 'wav' in file:
					os.system('cp ' + path + folder + '/' + file + ' data/hupa/' + quality + '/dates/dev' + date + '/' + args.n + '/')

	return sort(train_data, num_of_speakers), sort(dev_data, num_of_speakers)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input path; e.g., data/hupa/top_tier')
	parser.add_argument('--n', type = str, help = '1 (top tier) or 2(top tier)')
	parser.add_argument('--split', type = str, help = 'split method')
	parser.add_argument('--info', type = str, help = 'path + file of audio_info.txt')
	parser.add_argument('--lang', type = str, help = 'language')
	parser.add_argument('--speaker', default = 'same', help = 'different speakers or treating as the same')

	args = parser.parse_args()

	temp_audio_info_data = gather_audio_info(args.info)
	audio_info_data = []
	for tok in temp_audio_info_data:
		file = tok[0]
		file = file.split('_')
		if file[1] == args.n:
			audio_info_data.append(tok)

	quality = ''

	if args.n == '1':
		quality = 'top_tier'

	if args.n == '2':
		quality = 'second_tier'

	if not os.path.exists('data/hupa/' + quality + '/'):
		os.makedirs('data/hupa/' + quality + '/')

	### If doing random splits ###

	if args.split == 'random':

		if not os.path.exists('data/hupa/' + quality + '/random/'):
			os.makedirs('data/hupa/' + quality + '/random/')

		for i in range(11, 18):

			i = str(i)

			if not os.path.exists('data/hupa/' + quality + '/random/train' + str(i)):
				os.makedirs('data/hupa/' + quality + '/random/train' + str(i))

			if not os.path.exists('data/hupa/' + quality + '/random/dev' + str(i)):
				os.makedirs('data/hupa/' + quality + '/random/dev' + str(i))

			if not os.path.exists('data/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/')

			if not os.path.exists('data/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/')


			train_data, dev_data = random_split(args.input + 'all_text' + args.n + '.txt', args.n, audio_info_data, args.input + '/random/', str(i))

			train_f = ''

			with io.open('data/hupa/' + quality + '/random/train' + str(i) + '/text.' + args.n, 'w') as train_f:
				for tok in train_data:	
					train_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/random/train' + str(i) + '/text.* > data/hupa/' + quality + '/random/train' + str(i) + '/temp')
			os.system('mv data/hupa/' + quality + '/random/train' + str(i) + '/temp data/hupa/' + quality + '/random/train' + str(i) + '/text')
			
			corpus = text_to_corpus('data/hupa/' + quality + '/random/train' + str(i) + '/text.' + args.n)
		
			with io.open('data/hupa/' + quality + '/random/train' + str(i) + '/corpus.' + args.n, 'w', encoding = 'utf-8') as f:
				for tok in corpus:
					f.write(tok + '\n')

			dev_f = ''

			with io.open('data/hupa/' + quality + '/random/dev' + str(i) + '/text.' + args.n, 'w') as dev_f:
				for tok in dev_data:
					dev_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/random/dev' + str(i) + '/text.* > data/hupa/' + quality + '/random/dev' + str(i) + '/temp')
			os.system('mv data/hupa/' + quality + '/random/dev' + str(i) + '/temp data/hupa/' + quality + '/random/dev' + str(i) + '/text')
			
			with io.open('random_sort_data' + str(i) + '.sh', 'w') as f:
				for tok in train_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/' + '\n')

				for tok in dev_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/' + '\n')

			os.system('bash random_sort_data' + str(i) + '.sh')

			write_utt_spk('data/hupa/' + quality + '/' + args.split + '/', i, args.split, args.speaker, args.lang)


	if args.split == 'distance':

		if not os.path.exists('data/hupa/' + quality + '/distance/'):
			os.makedirs('data/hupa/' + quality + '/distance/')

		for i in range(1, 6):

			i = str(i)

			if not os.path.exists('data/hupa/' + quality + '/distance/train' + str(i)):
				os.makedirs('data/hupa/' + quality + '/distance/train' + str(i))

			if not os.path.exists('data/hupa/' + quality + '/distance/dev' + str(i)):
				os.makedirs('data/hupa/' + quality + '/distance/dev' + str(i))

			if not os.path.exists('data/hupa/' + quality + '/distance/train' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data/hupa/' + quality + '/distance/train' + str(i) + '/' + str(args.n) + '/')

			if not os.path.exists('data/hupa/' + quality + '/distance/dev' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data/hupa/' + quality + '/distance/dev' + str(i) + '/' + str(args.n) + '/')


			train_data, dev_data = split_with_wasserstein(audio_info_data, 'data/hupa/' + quality + '/distance/', i, args.n, args.lang)

			train_f = ''

			with io.open('data/hupa/' + quality + '/distance/train' + str(i) + '/text.' + args.n, 'w') as train_f:
				for tok in train_data:	
					train_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/distance/train' + str(i) + '/text.* > data/hupa/' + quality + '/distance/train' + str(i) + '/temp')
			os.system('mv data/hupa/' + quality + '/distance/train' + str(i) + '/temp data/hupa/' + quality + '/distance/train' + str(i) + '/text')
			
			corpus = text_to_corpus('data/hupa/' + quality + '/distance/train' + str(i) + '/text.' + args.n)
		
			with io.open('data/hupa/' + quality + '/distance/train' + str(i) + '/corpus.' + args.n, 'w', encoding = 'utf-8') as f:
				for tok in corpus:
					f.write(tok + '\n')

			dev_f = ''

			with io.open('data/hupa/' + quality + '/distance/dev' + str(i) + '/text.' + args.n, 'w') as dev_f:
				for tok in dev_data:
					dev_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/distance/dev' + str(i) + '/text.* > data/hupa/' + quality + '/distance/dev' + str(i) + '/temp')
			os.system('mv data/hupa/' + quality + '/distance/dev' + str(i) + '/temp data/hupa/' + quality + '/distance/dev' + str(i) + '/text')	

			with io.open('distance_sort_data' + str(i) + '.sh', 'w') as f:
				for tok in train_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/distance/train' + str(i) + '/' + str(args.n) + '/' + '\n')

				for tok in dev_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/distance/dev' + str(i) + '/' + str(args.n) + '/' + '\n')

			os.system('bash distance_sort_data' + str(i) + '.sh')

			write_utt_spk('data/hupa/' + quality + '/' + args.split + '/', i, args.split, args.speaker, args.lang)

	if args.split in ['len', 'ave_intensity', 'ave_pitch', 'ppl', 'num_word', 'word_type']:

		i = '1'

		if not os.path.exists('data/hupa/' + quality + '/' + args.split + '/'):
			os.makedirs('data/hupa/' + quality + '/' + args.split + '/')

		if not os.path.exists('data/hupa/' + quality + '/' + args.split + '/train' + str(i)):
			os.makedirs('data/hupa/' + quality + '/' + args.split + '/train' + str(i))

		if not os.path.exists('data/hupa/' + quality + '/' + args.split + '/dev' + str(i)):
			os.makedirs('data/hupa/' + quality + '/' + args.split + '/dev' + str(i))

		if not os.path.exists('data/hupa/' + quality + '/' + args.split + '/train' + str(i) + '/' + str(args.n) + '/'):
			os.makedirs('data/hupa/' + quality + '/' + args.split + '/train' + str(i) + '/' + str(args.n) + '/')

		if not os.path.exists('data/hupa/' + quality + '/' + args.split + '/dev' + str(i) + '/' + str(args.n) + '/'):
			os.makedirs('data/hupa/' + quality + '/' + args.split + '/dev' + str(i) + '/' + str(args.n) + '/')

		functions = {'random': random_split, 'len': duration_threshold_different, 'ave_intensity': average_intensity_threshold_different, 'ave_pitch': average_pitch_threshold_different, 'ppl': ppl_threshold_different, 'num_word': num_word_threshold_different, 'word_type': word_type_threshold_different, 'distance': split_with_wasserstein}
		function = functions[args.split]
		train_data, dev_data = function(audio_info_data, args.input + '/' + args.split + '/', '1', args.n, args.lang)

		train_f = ''

		with io.open('data/hupa/' + quality + '/' + args.split + '/train1' + '/text.' + args.n, 'w') as train_f:
			for tok in train_data:	
				train_f.write(tok + '\n')

		os.system('cat data/hupa/' + quality + '/' + args.split + '/train1' + '/text.* > data/hupa/' + quality + '/' + args.split + '/train1' + '/temp')
		os.system('mv data/hupa/' + quality + '/' + args.split + '/train1' + '/temp data/hupa/' + quality + '/' + args.split + '/train1' + '/text')
			
		corpus = text_to_corpus('data/hupa/' + quality + '/' + args.split + '/train1' + '/text.' + args.n)
		
		with io.open('data/hupa/' + quality + '/' + args.split + '/train1' + '/corpus.' + args.n, 'w', encoding = 'utf-8') as f:
			for tok in corpus:
				f.write(tok + '\n')

		dev_f = ''

		with io.open('data/hupa/' + quality + '/' + args.split + '/dev1' + '/text.' + args.n, 'w') as dev_f:
			for tok in dev_data:
				dev_f.write(tok + '\n')

		os.system('cat data/hupa/' + quality + '/' + args.split + '/dev1' + '/text.* > data/hupa/' + quality + '/' + args.split + '/dev1' + '/temp')
		os.system('mv data/hupa/' + quality + '/' + args.split + '/dev1' + '/temp data/hupa/' + quality + '/' + args.split + '/dev1' + '/text')

		with io.open('hupa_' + args.split + '_sort_data.sh', 'w') as f:
			for tok in train_data:
				file_name = tok.split()[0]
				audio_file = file_name + '.wav'

				f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/' + args.split + '/train1' + '/' + str(args.n) + '/' + '\n')

			for tok in dev_data:
				file_name = tok.split()[0]
				audio_file = file_name + '.wav'

				f.write('cp ' + args.input + audio_file + ' ' + 'data/hupa/' + quality + '/' + args.split + '/dev1' + '/' + str(args.n) + '/' + '\n')

		os.system('bash hupa_' + args.split + '_sort_data.sh')

		write_utt_spk('data/hupa/' + quality + '/' + args.split + '/', i, args.split, args.speaker, args.lang)

	if args.split == 'dates':

		if not os.path.exists('data/hupa/' + quality + '/dates/'):
			os.makedirs('data/hupa/' + quality + '/dates/')

		dates = all_dates(args.input)

		for date in dates:
			train_data, dev_data = date_split(args.input, date, args.n)
			train_f = ''
			with io.open('data/hupa/' + quality + '/dates/train' + date + '/text.' + args.n, 'w') as train_f:
				for tok in train_data:	
					train_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/dates/train' + date + '/text.* > data/hupa/' + quality + '/dates/train' + date + '/temp')
			os.system('mv data/hupa/' + quality + '/dates/train' + date + '/temp data/hupa/' + quality + '/dates/train' + date + '/text')

			corpus = text_to_corpus('data/hupa/' + quality + '/dates/train' + date + '/text.' + args.n)		
			with io.open('data/hupa/' + quality + '/dates/train' + date + '/corpus.' + args.n, 'w', encoding = 'utf-8') as f:
				for tok in corpus:
					f.write(tok + '\n')

			dev_f = ''
			with io.open('data/hupa/' + quality + '/dates/dev' + date + '/text.' + args.n, 'w') as dev_f:
				for tok in dev_data:
					dev_f.write(tok + '\n')

			os.system('cat data/hupa/' + quality + '/dates/dev' + date + '/text.* > data/hupa/' + quality + '/dates/dev' + date + '/temp')
			os.system('mv data/hupa/' + quality + '/dates/dev' + date + '/temp data/hupa/' + quality + '/dates/dev' + date + '/text')
		