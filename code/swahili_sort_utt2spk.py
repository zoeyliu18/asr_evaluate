import io, os, argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input path')
	parser.add_argument('--info', type = str, help = 'input path')
	
	args = parser.parse_args()

	info_data = {}
	with io.open(args.info, encoding = 'utf-8') as f:
		for line in f:
			if line.startswith('File') is False:
				toks = line.strip().split('\t')

				info_data[toks[0]] = toks[1]

	for folder in os.listdir(args.input):
		if folder.startswith('train') or folder.startswith('dev'):
			print(folder)
			with io.open('swahili_utt2spk.sh', 'w', encoding = 'utf-8') as f:
				f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
				f.write("for dir in " + args.input + folder + '\n')
				f.write('do' + '\n')
				f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
				f.write("	cat $dir/text | cut -d'_' -f1 > $dir/spk" + '\n')
				f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
				f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
				f.write('	rm $dir/utt $dir/spk' + '\n')
				f.write('done' + '\n')

			os.system('bash swahili_utt2spk.sh')

			for file in os.listdir(args.input + folder + '/'):
				if file == 'utt2spk':
					data = []
					print(args.input + folder + '/' + file)
					with io.open(args.input + folder + '/' + file, encoding = 'utf-8') as f:
						for line in f:
							toks = line.strip().split()
							data.append(toks)

					new_data = []
					for i in range(len(data)):
						toks = data[i]
						speaker = (toks[0].split('swahili')[1]).split('_')
						hour = speaker[1][0 : 2]
						date = speaker[-2]
						new_tok = [toks[0], 'SWH-' + hour + '-' + date]
					#	if toks[0].startswith('16'):
					#		speaker = toks[0].split('_')
					#		hour = speaker[2][0 : 2]
					#		date = speaker[-2]
					#		new_tok = [toks[0], 'SWH-' + hour + '-' + date]
					#	else:
					#		new_tok = toks
					#	assert new_tok[0] == toks[0]
						new_data.append(new_tok)

					with io.open(args.input + folder + '/' + 'utt2spk_temp', 'w', encoding = 'utf-8') as f:
						for tok in new_data:
							f.write(' '.join(w for w in tok) + '\n')

					os.system('mv ' + args.input + folder + '/' + 'utt2spk_temp ' + args.input + folder + '/' + 'utt2spk')
					os.system('utils/utt2spk_to_spk2utt.pl ' + args.input + folder + '/utt2spk | sort -k1 > ' + args.input + folder + '/spk2utt')

					with io.open(args.input + folder + '/' + 'wav.scp', 'w', encoding = 'utf-8') as f:
						for tok in data:
							new_tok = [tok[0], info_data[tok[0]]]
							f.write(' '.join(w for w in new_tok) + '\n')

					with io.open('swahili_compute_mfcc.sh', 'w', encoding = 'utf-8') as f:
						f.write("echo 'compute mfcc for train dev...'" + '\n')
						f.write("for dir in " + args.input + folder + '\n')
						f.write('do' + '\n')
						f.write('	bash /data/liuaal/kaldi/make_mfcc.sh --nj 4 $dir exp/make_mfcc/$dir $dir/mfcc' + '\n')
						f.write('	steps/compute_cmvn_stats.sh $dir exp/make_mfcc/$dir $dir/mfcc' + '\n')
						f.write('done')

					os.system('bash swahili_compute_mfcc.sh')

