import os, io

data = []
with open('iban/original/train1/old.scp') as f:
	for line in f:
		line = line.strip()
		line = line.replace('asr_iban/', '/data/liuaal/asr_data/Iban/')
		data.append(line)

with open('iban/original/train1/wav.scp', 'w') as f:
	for tok in data:
		f.write(tok + '\n')


#