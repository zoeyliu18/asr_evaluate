### Analysis of overlap between test sets given different data splits ###
### Using the first random split as the reference ###
import io, os, sys

splits = ''
if 'hupa' not in sys.argv[1]:
	splits = ['len_different', 'ave_pitch', 'ave_intensity', 'ppl', 'num_word', 'word_type', 'distance', 'random_different']
else:
	splits = ['len', 'ave_pitch', 'ave_intensity', 'ppl', 'num_word', 'word_type', 'distance', 'random']

ref = []
f = ''
if 'hupa' not in sys.argv[1]:
	f = io.open(sys.argv[1] + 'random_different/dev1/text')
else:
	f = io.open(sys.argv[1] + 'random/dev1/text')
for line in f:
	toks = line.strip().split()
	utt = toks[0]
	ref.append(utt)


for split in splits:
	for dev_set in os.listdir(sys.argv[1] + split + '/'):
		if dev_set.startswith('dev'):
			dev_set_utt = []
			with io.open(sys.argv[1] + split + '/' + dev_set + '/text') as f:
				for line in f:
					toks = line.strip().split()
					utt = toks[0]
					dev_set_utt.append(utt)

			c = 0
			for utt in dev_set_utt:
				if utt in ref:
					c += 1

			overlap_ratio = round(100 * c / len(dev_set_utt))

			print(split, dev_set)
			print(overlap_ratio)
			print('\n')

