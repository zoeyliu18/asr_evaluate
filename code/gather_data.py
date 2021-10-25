import io, os, argparse, random, librosa
from pydub import AudioSegment


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type = str, help = 'original training data path')
	parser.add_argument('--dev', type = str, help = 'original development data path')
#	parser.add_argument('--test', type = str, help = 'original test data path')
	parser.add_argument('--output', type = str, help = 'output path')
	
	args = parser.parse_args()

	train_path = args.train
	dev_path = args.dev 
#	test_path = args.test

	all_texts = {}
	all_info = []
	time = 0

	for path in [train_path, dev_path]: #, test_path]:
		for file in os.listdir(path):
			if file == 'text':
				with io.open(path + file, encoding = 'utf-8') as f:
					for line in f:
						toks = line.strip().split()
						if toks[0].startswith('16'):
							file_name = toks[0].split('_')
							hour = file_name[2][0 : 2]
							date = file_name[-2]
							audio_name = 'SWH-' + hour + '-' + date + '_' + '_'.join(w for w in file_name)
							all_texts[audio_name] = ' '.join(w for w in toks[1: ])
						else:
							all_texts[toks[0]] = ' '.join(w for w in toks[1: ])

	with io.open(args.output + 'audio_info.txt', 'w', encoding = 'utf-8') as outfile:
		header = ['File', 'Path', 'Duration', 'Transcript']
		outfile.write('\t'.join(w for w in header) + '\n')
		for path in [train_path, dev_path]:#, test_path]:
			for folder in os.listdir(path):
				if os.path.isdir(path + folder) is True:
					for file in os.listdir(path + folder):
						if file.endswith('.wav'):
							try:
					#	if file.endswith('.WAV') or file.endswith('.wav'):
					#	duration = len(AudioSegment.from_wav(path + folder + '/' + file))
								audio = file.split('.')[0]
								if '<UNK>' not in all_texts[audio] and '<music>' not in all_texts[audio] and audio.startswith('SWH'):
									duration = librosa.get_duration(filename = path + folder + '/' + file)

									time += duration
								#	audio_name = audio 
								#	if audio.startswith('16'):
								#		file_name = audio.split('_')
								#		hour = file_name[2][0 : 2]
								#		date = file_name[-2]

								#		audio_name = 'SWH-' + hour + '-' + date + '_' + '_'.join(w for w in file_name)
									outfile.write('\t'.join(str(w) for w in [audio, path + folder + '/' + file, duration, all_texts[audio]]) + '\n')
							except:
								pass

	print(time / (60 * 60))

