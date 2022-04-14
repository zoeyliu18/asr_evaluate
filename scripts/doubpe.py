data = []
with open('utt2dur') as f:
	for line in f:
		toks = line.split()[-1]
		data.append(float(toks))

sum(data)/3600


header = ['File', 'Path', 'Transcript', 'Duration', 'Pitch', 'Intensity', 'PPL', 'Num_word', 'Word_type', 'OOV', 'Speaker']

duration_dict = {}
pitch_dict = {}
intensity_dict = {}
ppl_dict = {}
num_word_dict = {}
word_type_dict = {}
with open('data/hupa/utterance_features.txt') as f:
	for line in f:
		if line.startswith('File') is False:
			toks = line.split('\t')
			file = toks[0]
			duration_dict[file] = float(toks[3])
			pitch_dict[file] = float(toks[4])
			intensity_dict[file] = float(toks[5])
			ppl_dict[file] = float(toks[6])
			num_word_dict[file] = int(toks[7])
			word_type_dict[file] = int(toks[8])


data = []

train_duration = []
dev_duration = []

with open('data/hupa/top_tier/len/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_duration.append(duration_dict[toks[0]])

with open('data/hupa/top_tier/len/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_duration.append(duration_dict[toks[0]])

print(sum(data) / 3600)
train_duration.sort()
dev_duration.sort()

print(train_duration[-1])
print(dev_duration[0])


##########


data = []

train_pitch = []
dev_pitch = []

with open('data/hupa/top_tier/ave_pitch/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_pitch.append(pitch_dict[toks[0]])

with open('data/hupa/top_tier/ave_pitch/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_pitch.append(pitch_dict[toks[0]])

print(sum(data) / 3600)
train_pitch.sort()
dev_pitch.sort()

print(train_pitch[-1])
print(dev_pitch[0])


##########


data = []

train_intensity = []
dev_intensity = []

with open('data/hupa/top_tier/ave_intensity/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_intensity.append(intensity_dict[toks[0]])

with open('data/hupa/top_tier/ave_intensity/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_intensity.append(intensity_dict[toks[0]])

print(sum(data) / 3600)
train_intensity.sort()
dev_intensity.sort()

print(train_intensity[-1])
print(dev_intensity[0])


##########


data = []

train_ppl = []
dev_ppl = []

with open('data/hupa/top_tier/ppl/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_ppl.append(ppl_dict[toks[0]])

with open('data/hupa/top_tier/ppl/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_ppl.append(ppl_dict[toks[0]])

print(sum(data) / 3600)
train_ppl.sort()
dev_ppl.sort()

print(train_ppl[-1])
print(dev_ppl[0])

##########


data = []

train_num_word = []
dev_num_word = []

with open('data/hupa/top_tier/num_word/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_num_word.append(num_word_dict[toks[0]])

with open('data/hupa/top_tier/num_word/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_num_word.append(num_word_dict[toks[0]])

print(sum(data) / 3600)
train_num_word.sort()
dev_num_word.sort()

print(train_num_word[-1])
print(dev_num_word[0])

##########

data = []

train_word_type = []
dev_word_type = []

with open('data/hupa/top_tier/word_type/train1/utt2dur') as f:
	for line in f:
		toks = line.split()
		train_word_type.append(word_type_dict[toks[0]])

with open('data/hupa/top_tier/word_type/dev1/utt2dur') as f:
	for line in f:
		toks = line.split()
		data.append(float(toks[-1]))
		dev_word_type.append(word_type_dict[toks[0]])

print(sum(data) / 3600)
train_word_type.sort()
dev_word_type.sort()

print(train_word_type[-1])
print(dev_word_type[0])

