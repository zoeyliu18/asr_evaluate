import io, os, argparse
from pydub import AudioSegment

def remove(character_list, character1, character2):

	if character1 in character_list:
	
		assert character2 in character_list
	
		index_list = []
		new_character = ''
		start = [i for i, x in enumerate(character_list) if x == character1]
		end = [i for i, x in enumerate(character_list) if x == character2]
	
		assert len(start) == len(end)
	
		if len(list(zip(start, end))) == 1:
			tok = list(zip(start, end))[0]
			new_character += ''.join(c for c in character_list[ : tok[0]])
			new_character += ''.join(c for c in character_list[tok[1] + 1 : ])
	
		else:
			for i in range(len(list(zip(start, end)))):
				tok = list(zip(start, end))[i]
			
				try:
					next_tok = list(zip(start, end))[i + 1]
					new_character += ''.join(c for c in character_list[ : tok[0]])
					new_character += ''.join(c for c in character_list[tok[1] + 1 : next_tok[0]])
			
				except:
					print('no more elements')
			
			new_character += ''.join(c for c in character_list[list(zip(start, end))[-1][1] + 1: ])
		
		return list(new_character)

	else:
		return character_list


def split_wav(audio_file, file_id, path, output, num_of_speakers):
	
	audio = AudioSegment.from_wav(path + audio_file)
	audio = audio.set_frame_rate(16000)
	file_name = audio_file.split('.')[0]
	label = file_name[3 : ]
	transcription_file = file_name + '.txt'
	
	corpus = [] ## for training language model later
	data = []
	list_of_timestamps = []

	with io.open(path + '/' + transcription_file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')

			if toks[0].startswith('Begin') is False:
				if toks[-1].startswith('{') and toks[-1].endswith('}'):
					print('FULL ENGLISH UTTERANCE')
					print(audio_file)
					print(toks[-1])
					print('\n')
					print('\n')

				### not including utterances that are fully English

				else:
					if toks != ['']:
						data.append(toks)
						text = toks[-1].replace('{', '')
						text = text.replace('}', '')
						text = text.replace('!', '')
						text = text.replace('.', '')
						text = text.replace('?', '')
						text = text.replace('/', '')
						text = text.replace('[', '')
						text = text.replace(']', '')
				
						new_text = []
				
						for w in text.split():
							w = list(w)
							new_w = remove(w, '(', ')')
							new_w = ''.join(c for c in new_w)
							new_text.append(new_w)

					#	corpus.append(' '.join(w for w in new_text))


	for i in range(len(data)):
		list_of_timestamps.append([float(data[i][0]), float(data[i][1])])

#	start = list_of_timestamps[0] * 1000

#	list_of_timestamps = list_of_timestamps[1 : ]
#	list_of_timestamps.append(float(data[-1][1]))

	text = []

	assert len(list_of_timestamps) == len(data)

	for idx in range(len(list_of_timestamps)):
		tok = list_of_timestamps[idx]

		start = tok[0] * 1000
		end = tok[1] * 1000

#	for idx, t in enumerate(list_of_timestamps):
#		if idx == len(list_of_timestamps):
#			break
		
#		end = t * 1000

		new_file_id = ''

		if len(str(file_id)) == 1:
			new_file_id = '000' + str(file_id)
		if len(str(file_id)) == 2:
			new_file_id = '00' + str(file_id)
		if len(str(file_id)) == 3:
			new_file_id = '0' + str(file_id)
		if len(str(file_id)) == 4:
			new_file_id = str(file_id)

		if start > end:
			print('Starting time is later than end time')
			print(start)
			print(end)
			print(audio_file)
			print('\n')
			print('\n')

		if '{' in data[idx][-1]:
			print(audio_file + '   Have English utterances')
			print(data[idx][-1])
			print('\n')
			print('\n')
		
		new_transcript = []

		for w in data[idx][-1].split():
			w = list(w)
			new_w = remove(w, '(', ')')
			new_w = ''.join(c for c in new_w)
			new_w = new_w.replace('{', '')
			new_w = new_w.replace('}', '')
			new_w = new_w.replace('!', '')
			new_w = new_w.replace('.', '')
			new_w = new_w.replace('?', '')
			new_w = new_w.replace('/', '')
			new_w = new_w.replace('[', '')
			new_w = new_w.replace(']', '')

			new_transcript.append(new_w)
			if '(' in w:
				print(w)
				print(new_w)

		audio_chunk = audio[start:end]
		audio_chunk.export(output + '/' + "verdena_" + num_of_speakers +  "_" + new_file_id + ".wav", format = "wav")	

		new_transcript = ' '.join(w for w in new_transcript)

		corpus.append(new_transcript)

		text.append("verdena_" + num_of_speakers +  "_" + new_file_id + ' ' + new_transcript + ' ' + str(start) + ' ' + str(end))
	
		start = end #pydub works in millisec
		file_id += 1

		if new_file_id == '':
			print('NO FILE ID')
			print(file_id)
			print(audio_file)

	return corpus, text, file_id

def order_file(path):

	dates_list = []
	ordered_dates_list = []

	days_list = []
	months_list = []
	years_list = []

	for file in os.listdir(path):
		if file.endswith('WAV'):
			dates = file.split('.')[0].split('-')[1]
			dates_list.append(dates)

			day = dates[ : 2]
			month = dates[2 : 4]
			year = dates[-2 : ]

			if day[0] == '0':
				day = day[1]

			if month[0] == '0':
				month = month[1]

			if year[0] == '0':
				year = year[1]

			days_list.append(int(day))
			months_list.append(int(month))
			years_list.append(int(year))

	days_list = list(set(days_list))
	months_list = list(set(months_list))
	years_list = list(set(years_list))

	days_list.sort()
	months_list.sort()
	years_list.sort()


	for y in years_list:
		if len(str(y)) == 1:
			y = '0' + str(y)

		for m in months_list:			
			if len(str(m)) == 1:
				m = '0' + str(m)
			
			for d in days_list:
				if len(str(d)) == 1:
					d = '0' + str(d)

				date = str(d) + str(m) + str(y)

				if date in dates_list:
					ordered_dates_list.append(date)

	return ordered_dates_list


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input path')
	parser.add_argument('--output', type = str, help = 'output path')
	parser.add_argument('--n', type = str, help = '1 (top tier) or 2(top tier)')
	parser.add_argument('--start', type = str, help = 'number after last training data')

	args = parser.parse_args()

	ordered_dates_list = order_file(args.input)

	### Generating all data ###

	file_id_start = int(args.start)

	all_corpus = io.open(args.output + '/' + 'corpus' + args.n + '.txt', 'w') ### Just texts from transcriptions
	all_text = io.open(args.output + '/' + 'all_text' + args.n + '.txt', 'w')

	all_files = []

	for file in os.listdir(args.input + '/'):
		print(file)
		if file.endswith('.WAV'):
			file_name = file.split('.')[0]
			transcription_file = file_name + '.txt'
			if transcription_file in os.listdir(args.input):
				all_files.append(file)

	for file in all_files:
		corpus, text, file_id_start = split_wav(file, file_id_start, args.input + '/', args.output + '/', args.n)
		for tok in corpus:
			all_corpus.write(tok + '\n')
		for tok in text:
			all_text.write(tok + '\n')

	### Generating data for each date ###

	file_id_start = int(args.start)

	for i in range(len(ordered_dates_list)):

		dates = ordered_dates_list[i]
		print(dates)
	
		if not os.path.exists(args.input + dates + '/'):
			os.makedirs(args.input + dates + '/')
			os.system('cp ' + args.input + '*' + dates + '-* ' + args.input + dates + '/')
		#	print('mv ' + args.input + '*' + dates + '-* ' + args.input + dates + '/')

			if not os.path.exists(args.output + dates + '/'):
				os.makedirs(args.output + dates + '/')

			all_corpus = io.open(args.output + dates + '/' + 'corpus' + args.n + '.txt', 'w') ### Just texts from transcriptions
			all_text = io.open(args.output + dates + '/' + 'all_text' + args.n + '.txt', 'w')

			all_files = []

			for file in os.listdir(args.input + dates + '/'):
				print(file)
				if file.endswith('.WAV'):
					file_name = file.split('.')[0]
					transcription_file = file_name + '.txt'
					if transcription_file in os.listdir(args.input):
						all_files.append(file)

			for file in all_files:
				corpus, text, file_id_start = split_wav(file, file_id_start, args.input + dates + '/', args.output + dates + '/', args.n)
				for tok in corpus:
					all_corpus.write(tok + '\n')
				for tok in text:
					all_text.write(tok + '\n')


