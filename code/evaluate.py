import io, os, argparse, statistics
from jiwer import wer

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'ASR model output')
	parser.add_argument('--state', type = str, help = 'whethere calculating or collecting WER/CER results')
#	parser.add_argument('--lang', type = str, help = 'language')

	args = parser.parse_args()

	if args.state == 'c':
		for lang_dir in os.listdir(args.input):
			if lang_dir in ['fongbe', 'swahili']: #'wolof', 'iban',
				for evaluate_dir in os.listdir(args.input + lang_dir + '/'):
					for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
						if 'RESULTS' not in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
							index = output_dir[6 : ]

							gold_dict = {}

							gold = []
							pred = []

							gold_word = []
							pred_word = []

							with io.open('data/' + lang_dir + '/' + evaluate_dir + '/dev' + index + '/text', encoding = 'utf-8') as f:
								for line in f:
									utterance = line.strip().split()
									toks = utterance[1 : ]
									gold_dict[utterance[0]] = toks

									for w in toks:
										gold_word.append(' '.join(c for c in w))
									toks = ''.join(w for w in toks)
									gold.append(' '.join(c for c in toks))

							wers = []
							evaluations = []

							if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
								with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									for line in f:
										toks = line.split()
										wers.append(toks[1])
										evaluations.append(toks)

							else:
								with io.open('evaluate.sh', 'w', encoding = 'utf-8') as eval:
									eval.write('for x in ' + args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > ' + args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS')
									eval.write('\n')

								os.system('bash evaluate.sh')

								with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									for line in f:
										toks = line.split()
										wers.append(toks[1])
										evaluations.append(toks)

							if len(wers) != 14:
								print('Output not complete ' + args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir)
								pass

							else:
								min_wer = min(wers)
								min_wer_idx = wers.index(min_wer)

								best_d = evaluations[min_wer_idx][-1].split('/')[: -1]
								best_d = '/'.join(w for w in best_d)

								with io.open(best_d + '/log/decode.1.log', encoding = 'utf-8') as f:
									for line in f:
										try:
											utterance = line.strip().split()
											if utterance[0] in gold_dict:
												toks = utterance[1: ]
												for w in toks:
													pred_word.append(' '.join(c for c in w))
												toks = ''.join(w for w in toks)
												pred.append(' '.join(c for c in toks))
										except:
											pass

						#		with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/cer_RESULTS', 'w', encoding = 'utf-8') as f:
						#			f.write('CER: ' + str(round(wer(gold, pred) * 100 / 2)) + '\n')
						#			f.write('CER word level: ' + str(round(wer(gold_word, pred_word) * 100 / 2)) + '\n')

			if lang_dir == 'hupa':
				for quality in ['top_tier', 'second_tier']:
					for evaluate_dir in os.listdir(args.input + lang_dir + '/' + quality + '/'):
						for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/'):
							if 'RESULTS' not in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/'):
								index = output_dir[6 : ]

								gold_dict = {}

								gold = []
								pred = []

								gold_word = []
								pred_word = []

								with io.open('data/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/dev' + index + '/text', encoding = 'utf-8') as f:
									for line in f:
										utterance = line.strip().split()
										toks = utterance[1 : ]
										gold_dict[utterance[0]] = toks

										for w in toks:
											gold_word.append(' '.join(c for c in w))
										toks = ''.join(w for w in toks)
										gold.append(' '.join(c for c in toks))

								wers = []
								evaluations = []

								if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/'):
									with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
										for line in f:
											toks = line.split()
											wers.append(toks[1])
											evaluations.append(toks)

								else:
									with io.open('evaluate.sh', 'w', encoding = 'utf-8') as eval:
										eval.write('for x in ' + args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > ' + args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS')
										eval.write('\n')

									os.system('bash evaluate.sh')

									with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
										for line in f:
											toks = line.split()
											wers.append(toks[1])
											evaluations.append(toks)

								if len(wers) != 14:
									print('Output not complete ' + args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir)
									pass

								else:
									min_wer = min(wers)
									min_wer_idx = wers.index(min_wer)

									best_d = evaluations[min_wer_idx][-1].split('/')[: -1]
									best_d = '/'.join(w for w in best_d)

									with io.open(best_d + '/log/decode.1.log', encoding = 'utf-8') as f:
										for line in f:
											try:
												utterance = line.strip().split()
												if utterance[0] in gold_dict:
													toks = utterance[1: ]
													for w in toks:
														pred_word.append(' '.join(c for c in w))
													toks = ''.join(w for w in toks)
													pred.append(' '.join(c for c in toks))
											except:
												pass

								#	with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/cer_RESULTS', 'w', encoding = 'utf-8') as f:
								#		f.write('CER: ' + str(round(wer(gold, pred) * 100 / 2)) + '\n')
								#		f.write('CER word level: ' + str(round(wer(gold_word, pred_word) * 100 / 2)) + '\n')

	if args.state == 'a':
		for lang_dir in os.listdir(args.input):
			if lang_dir in ['fongbe', 'swahili', 'wolof', 'iban']:
				for evaluate_dir in os.listdir(args.input + lang_dir + '/'):

					all_wers = []
					all_cers = []

					for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
						wers = []
						cers = []

						if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
							with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
								for line in f:
									toks = line.split()
									wers.append(float(toks[1]))

							if wers == []:
								print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
							elif len(wers) != 14:
								print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' Results Not Complete')
							else:
								all_wers.append(min(wers))
						else:
							print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' No Results')

					if all_wers != []:
						if evaluate_dir in ['random_different', 'distance', 'heldout_speaker']:
							print(lang_dir + '\t' + evaluate_dir + '\t' + str(statistics.mean(all_wers)) + '\t' + str(len(all_wers)) + '\t' + str(statistics.stdev(all_wers)) + '\t' +  str(max(all_wers) - min(all_wers)))
						else:
							print(lang_dir + '\t' + evaluate_dir + '\t' + str(statistics.mean(all_wers)) + '\t' + str(len(all_wers)))

			if lang_dir in ['hupa']: #'wolof', 'iban',
				for quality in ['top_tier', 'second_tier']:
					for evaluate_dir in os.listdir(args.input + lang_dir + '/' + quality + '/'):
						all_wers = []
						all_cers = []

						for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/'):
							wers = []
							cers = []

							if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/'):
								with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									for line in f:
										toks = line.split()
										wers.append(float(toks[1]))

								if wers == []:
									print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
								elif len(wers) != 14:
									print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' Results Not Complete')
								else:
									all_wers.append(min(wers))
							else:
								print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' No Results')

						if all_wers != []:
							if evaluate_dir in ['random', 'distance', 'dates']:
								print(lang_dir + '\t' + quality + '\t' + evaluate_dir + '\t' + str(round(statistics.mean(all_wers), 2)) + '\t' + str(len(all_wers)) + '\t' + str(round(statistics.stdev(all_wers), 2)) + '\t' +  str(round(max(all_wers) - min(all_wers), 2)))
							else:
								print(lang_dir + '\t' + quality + '\t' + evaluate_dir + '\t' + str(round(statistics.mean(all_wers), 2)) + '\t' + str(len(all_wers)))

	#		with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/cer_RESULTS', encoding = 'utf-8') as f:
	#			for line in f:
	#				if line.startswith('CER: '):
	#					toks = line.split()
	#					cers.append(float(toks[1]))

	#		all_cers.append(min(cers))


	#	print(statistics.mean(all_cers))
