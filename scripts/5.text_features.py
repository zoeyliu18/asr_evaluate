### python3 script/5.text_features.py --lm data/hupa/local/tmp/lm.arpa --info data/hupa/dates_audio_info.txt --lang hupa --output data/hupa/


import io, os, argparse


def gather_audio_info(file):

	data = []

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')
			if toks[0].startswith('File') is False or toks[0] != 'File':
				data.append(toks)

	return data



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--lm', type = str, help = 'language model')
	parser.add_argument('--info', type = str, help = 'audio_info.txt file')
	parser.add_argument('--lang', type = str, help = 'language')
	parser.add_argument('--output', type = str, help = 'output path')
	
	args = parser.parse_args()

	data = gather_audio_info(args.info)

	all_texts = []
	num_word_list = []
	word_type_list = []
	ppl_list = []

	for tok in data:
		text = ''
		if args.lang not in ['hupa']:
			all_texts.append(tok[-1])
			text = tok[-1].split()
		else:
			text = tok[-1].split()[ : -2]
			all_texts.append(' '.join(w for w in text))
		
		num_word_list.append(len(text))
		word_type_list.append(len(set(text)))

	with io.open('temp', 'w', encoding = 'utf-8') as f:
		for text in all_texts:
			f.write(text + '\n')

	os.system('ngram -lm ' + args.lm + ' -ppl temp -debug 1 > ' + args.lang + '_lm.txt')

	with io.open(args.lang + '_lm.txt', encoding = 'utf-8') as f:
		for line in f:
			if line.startswith('0 zeroprobs'):
				toks = line.split()
				ppl = toks[-3]
				ppl_list.append(ppl)
	print(len(all_texts))
	print(len(data))
	print(len(ppl_list))

	if 'dates' not in args.info:
		outfile = io.open(args.output + args.lang + '_lm_info.txt', 'w', encoding = 'utf-8')
		header = ['File', 'Path', 'Duration', 'Transcript', 'PPL', 'Num_word', 'Word_type']
		outfile.write('\t'.join(w for w in header) + '\n')
		for i in range(len(data)):
			tok = data[i]
			tok.append(ppl_list[i])
			tok.append(num_word_list[i])
			tok.append(word_type_list[i])
			outfile.write('\t'.join(str(w) for w in tok) + '\n')

	else:
		outfile = io.open(args.output + args.lang + '_dates_lm_info.txt', 'w', encoding = 'utf-8')
		header = ['File', 'Path', 'Duration', 'Transcript', 'PPL', 'Num_word', 'Word_type']
		outfile.write('\t'.join(w for w in header) + '\n')
		for i in range(len(data)):
			tok = data[i]
			tok.append(ppl_list[i])
			tok.append(num_word_list[i])
			tok.append(word_type_list[i])
			outfile.write('\t'.join(str(w) for w in tok) + '\n')





