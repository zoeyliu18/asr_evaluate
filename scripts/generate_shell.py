import io, os

data = []
with io.open('iban_random10.sh') as f:
	for line in f:
		data.append(line.strip())

for i in range(11, 24):
	print(i)
	new_data = []
	for tok in data:
		new_tok = tok
		new_tok = new_tok.replace('spk10', 'spk' + str(i))
		new_tok = new_tok.replace('mfcc10', 'mfcc' + str(i))
		new_tok = new_tok.replace('train10', 'train' + str(i))
		new_tok = new_tok.replace('dev10', 'dev' + str(i))
		new_tok = new_tok.replace('system10', 'system' + str(i))
		new_data.append(new_tok)
	with io.open('iban_random' + str(i) + '.sh', 'w') as f:
		for tok in new_data:
			f.write(tok + '\n')

	
pbs_data = []
with io.open('iban_random10.pbs') as f:
	for line in f:
		pbs_data.append(line.strip())

for i in range(11, 24):
	new_data = []
	for tok in pbs_data:
		new_data.append(tok.replace('random10', 'random' + str(i)))
	with io.open('iban_random' + str(i) + '.pbs', 'w') as f:
		for tok in new_data:
			f.write(tok + '\n')
