### python3 script/1.hupa_audio_info.py --input data/ --output data/hupa/

import io, os, argparse, random, librosa
from pydub import AudioSegment


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input to data/')
	parser.add_argument('--output', type = str, help = 'output path')
	
	args = parser.parse_args()

	top_tier_path = args.input + 'hupa/top_tier'
	second_tier_path = args.input + 'hupa/second_tier'


	all_random_texts = {}
	all_info = []
	time = 0

	for path in [top_tier_path, second_tier_path]:
		for file in os.listdir(path):
			if 'all_text' in file:
				with io.open(path + '/' + file) as f:
					for line in f:
						line = line.strip().split()
						file_name = line[0]
						text = line[1 : ]
						all_random_texts[file_name] = ' '.join(w for w in text)

	with io.open(args.output + 'audio_info.txt', 'w', encoding = 'utf-8') as outfile:
		header = ['File', 'Path', 'Duration', 'Transcript']
		outfile.write('\t'.join(w for w in header) + '\n')
		date = ''
		for path in [top_tier_path, second_tier_path]:
			temp_quality = path.split('/')[-1]
			quality = ''
			if temp_quality == 'top_tier':
				quality = '1'
			if temp_quality == 'second_tier':
				quality = '2'
			for file in os.listdir(path):
				if file.endswith('.wav'):
					try:
						audio = file.split('.')[0]
						if '<UNK>' not in all_random_texts[audio] and '<music>' not in all_random_texts[audio] and audio.startswith('verdena'):
							duration = librosa.get_duration(filename = path + '/' + '/' + file)

							time += duration
							outfile.write('\t'.join(str(w) for w in [audio, path + '/' + file, duration, all_random_texts[audio]]) + '\n')
					except:
						pass


	all_dates_texts = {}
	all_info = []
	time = 0

	for path in [top_tier_path, second_tier_path]:
		for folder in os.listdir(path):
			if os.path.isdir(path + '/' + folder) is True and len(folder) == 6 and folder[0].isdigit() is True:
				for file in os.listdir(path + '/' + folder):
					if 'all_text' in file:
						with io.open(path + '/' + folder + '/' + file) as f:
							for line in f:
								line = line.strip().split()
								file_name = line[0]
								text = line[1 : ]
								all_dates_texts[file_name] = ' '.join(w for w in text)


	with io.open(args.output + 'dates_audio_info.txt', 'w', encoding = 'utf-8') as outfile:
		header = ['File', 'Path', 'Duration', 'Transcript']
		outfile.write('\t'.join(w for w in header) + '\n')
		for path in [top_tier_path, second_tier_path]:
			temp_quality = path.split('/')[-1]
			quality = ''
			if temp_quality == 'top_tier':
				quality ='1'
			if temp_quality == 'second_tier':
				quality = '2'
			for folder in os.listdir(path):
				if os.path.isdir(path + '/' + folder) is True and len(folder) == 6 and folder[0].isdigit() is True:
					print(folder)
					date = folder

					for file in os.listdir(path + '/' + folder):
						if file.endswith('.wav'):
							try:
								audio = file.split('.')[0]
								if '<UNK>' not in all_dates_texts[audio] and '<music>' not in all_dates_texts[audio] and audio.startswith('verdena'):
									duration = librosa.get_duration(filename = path + '/' + folder + '/' + file)

									time += duration
									outfile.write('\t'.join(str(w) for w in [audio, path + '/' + folder + '/' + file, duration, all_dates_texts[audio]]) + '\n')
							except:
								pass
