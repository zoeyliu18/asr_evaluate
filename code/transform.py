import io, os, argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'original training data path')
	parser.add_argument('--output', type = str, help = 'original development data path')
	
	args = parser.parse_args()

	os.system('cat ' + args.input + 'text > ' + args.input + 'temp')

	data = []
	with io.open(args.input + 'temp') as f:
		for line in f:
			toks = line.strip().split()
			new_tok = toks[ : -2]
			data.append(new_tok)

	with io.open(args.input + 'text', 'w') as f:
		for tok in data:
			f.write(' '.join(w for w in tok) + '\n')


