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
			if lang_dir in ['fongbe', 'swahili', 'wolof', 'iban']:
				for evaluate_dir in os.listdir(args.input + lang_dir + '/'):
					for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
						if 2 > 1:
					#	if 'RESULTS' not in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/') or os.stat(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS').st_size == 0:
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

							if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/') and os.stat(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS').st_size != 0:
								with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									for line in f:
										toks = line.split()
										wers.append(toks[1])
										evaluations.append(toks)

							if len(evaluations) != 14:
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
								
								best_d = args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/'					

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

			if lang_dir in ['hupa', 'hupa_with_trans']:
				for quality in ['top_tier', 'second_tier']:
					for evaluate_dir in os.listdir(args.input + lang_dir + '/' + quality + '/'):
						for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/'):
							if 2 > 1:
						#	if 'RESULTS' not in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/') or os.stat(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS').st_size == 0:
								index = output_dir[6 : ]

								gold_dict = {}

								gold = []
								pred = []

								gold_word = []
								pred_word = []

								with io.open('data/' + 'hupa' + '/' + quality + '/' + evaluate_dir + '/dev' + index + '/text', encoding = 'utf-8') as f:
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

								if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/') and os.stat(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS').st_size != 0:
									with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
										for line in f:
											toks = line.split()
											wers.append(toks[1])
											evaluations.append(toks)

								if len(evaluations) != 14:
									print(len(evaluations))
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
									print(len(wers))
									print('Output not complete ' + args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir)
									pass

								else:
									
									best_d = args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/'					

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

	if args.state == 'a':
		for lang_dir in os.listdir(args.input):
			if lang_dir in ['fongbe', 'wolof', 'iban', 'swahili']:
				all_wers_results = {}
				all_heldout_wers = {}
				all_heldout_duration = {}
				all_random_wers = {}
				all_random_duration = {}
				all_distance_wers = {}
				all_distance_duration = {}

				for evaluate_dir in os.listdir(args.input + lang_dir + '/'):

					all_wers = []
					all_cers = []

					for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
						wers = []
						cers = []

						if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
							with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
								for line in f:
									if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
										toks = line.split()
										wers.append(float(toks[1]))

							if wers == []:
								print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
							else:
								all_wers.append(min(wers))

								if evaluate_dir == 'heldout_speaker':
									all_heldout_wers[output_dir[6 : ]] = min(wers)
								if evaluate_dir == 'random_different':
									all_random_wers[output_dir[6 : ]] = min(wers)
								if evaluate_dir == 'distance':
									all_distance_wers[output_dir[6 : ]] = min(wers)
						else:
							print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' No Results')

					if all_wers != []:
						if evaluate_dir in ['random_different', 'distance', 'heldout_speaker']:
							print(lang_dir + '\t' + evaluate_dir + '\t' + str(statistics.mean(all_wers)) + '\t' + str(len(all_wers)) + '\t' + str(statistics.stdev(all_wers)) + '\t' +  str(max(all_wers) - min(all_wers)))
						else:
							print(lang_dir + '\t' + evaluate_dir + '\t' + str(statistics.mean(all_wers)) + '\t' + str(all_wers[0]))
							all_wers_results[evaluate_dir] = statistics.mean(all_wers)

				for k, v in all_heldout_wers.items():
					dev_duration = 0
					with io.open('data/' + lang_dir + '/heldout_speaker/dev' + k + '/utt2dur') as f:
						for line in f:
							toks = line.strip().split()
							dev_duration += float(toks[1])
					all_heldout_duration[k] = dev_duration

				for k, v in all_random_wers.items():
					dev_duration = 0
					with io.open('data/' + lang_dir + '/random_different/dev' + k + '/utt2dur') as f:
						for line in f:
							toks = line.strip().split()
							dev_duration += float(toks[1])
					all_random_duration[k] = dev_duration

				for k, v in all_distance_wers.items():
					dev_duration = 0
					with io.open('data/' + lang_dir + '/distance/dev' + k + '/utt2dur') as f:
						for line in f:
							toks = line.strip().split()
							dev_duration += float(toks[1])
					all_distance_duration[k] = dev_duration

				with io.open(lang_dir + '_eval.txt', 'w') as f:
					f.write('\t'.join(w for w in ['Language', 'Speaker', 'Duration', 'WER', 'Evaluation']) + '\n')
					for k, v in all_heldout_duration.items():
						info = ''
						if lang_dir != 'swahili':
							info = [lang_dir, k, v, all_heldout_wers[k], 'heldout_speaker']
						else:
							info = [lang_dir, k, v, all_heldout_wers[k], 'heldout_session']
						f.write('\t'.join(str(w) for w in info) + '\n')
					for k, v in all_random_duration.items():
						info = [lang_dir, k, v, all_random_wers[k], 'random_different']
						f.write('\t'.join(str(w) for w in info) + '\n')
					for k, v in all_distance_duration.items():
						info = [lang_dir, k, v, all_distance_wers[k], 'distance']
						f.write('\t'.join(str(w) for w in info) + '\n')
					for k, v in all_wers_results.items():
						print(k, v)
						info = [lang_dir, '', '', v, k]
						f.write('\t'.join(str(w) for w in info) + '\n')

			if lang_dir in ['hupa']:
				for quality in ['top_tier', 'second_tier']:
					all_wers_results = {}
					all_heldout_wers = {}
					all_heldout_duration = {}
					all_random_wers = {}
					all_random_duration = {}
					all_distance_wers = {}
					all_distance_duration = {}

					for evaluate_dir in os.listdir(args.input + lang_dir + '/' + quality + '/'):
						all_wers = []
						all_cers = []

						for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/'):
							wers = []
							cers = []

							if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/'):
								with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									for line in f:
										if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
											toks = line.split()
											wers.append(float(toks[1]))

								if wers == []:
									print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
								else:
									all_wers.append(min(wers))

									if evaluate_dir == 'dates':
										all_heldout_wers[output_dir[6 : ]] = min(wers)
									if evaluate_dir == 'random':
										all_random_wers[output_dir[6 : ]] = min(wers)
									if evaluate_dir == 'distance':
										all_distance_wers[output_dir[6 : ]] = min(wers)
							else:
								print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' No Results')

						if all_wers != []:
							if evaluate_dir in ['random', 'distance', 'dates']:
								try:
									print(lang_dir + '\t' + quality + '\t' + evaluate_dir + '\t' + str(round(statistics.mean(all_wers), 2)) + '\t' + str(len(all_wers)) + '\t' + str(round(statistics.stdev(all_wers), 2)) + '\t' +  str(round(max(all_wers) - min(all_wers), 2)))
								except:
									print(lang_dir + '\t' + quality + '\t' + evaluate_dir + '\t' + str(len(all_wers)))
							else:
								print(lang_dir + '\t' + quality + '\t' + evaluate_dir + '\t' + str(all_wers[0]))
								all_wers_results[evaluate_dir] = statistics.mean(all_wers)

					for k, v in all_heldout_wers.items():
						dev_duration = 0
						with io.open('data/' + 'hupa' + '/' + quality + '/dates/dev' + k + '/utt2dur') as f:
							for line in f:
								toks = line.strip().split()
								dev_duration += float(toks[1])
						all_heldout_duration[k] = dev_duration

					for k, v in all_random_wers.items():
						dev_duration = 0
						with io.open('data/' + 'hupa' + '/' + quality + '/random/dev' + k + '/utt2dur') as f:
							for line in f:
								toks = line.strip().split()
								dev_duration += float(toks[1])
						all_random_duration[k] = dev_duration

					for k, v in all_distance_wers.items():
						dev_duration = 0
						with io.open('data/' + 'hupa' + '/' + quality + '/distance/dev' + k + '/utt2dur') as f:
							for line in f:
								toks = line.strip().split()
								dev_duration += float(toks[1])
						all_distance_duration[k] = dev_duration

					with io.open(lang_dir + '_' + quality + '_eval.txt', 'w') as f:
						f.write('\t'.join(w for w in ['Language', 'Speaker', 'Duration', 'WER', 'Evaluation']) + '\n')
						for k, v in all_heldout_duration.items():
							info = [lang_dir, k, v, all_heldout_wers[k], 'heldout_session']
							f.write('\t'.join(str(w) for w in info) + '\n')
						for k, v in all_random_duration.items():
							info = [lang_dir, k, v, all_random_wers[k], 'random_different']
							f.write('\t'.join(str(w) for w in info) + '\n')
						for k, v in all_distance_duration.items():
							info = [lang_dir, k, v, all_distance_wers[k], 'distance']
							f.write('\t'.join(str(w) for w in info) + '\n')
						for k, v in all_wers_results.items():
							info = [lang_dir, '', '', v, k]
							f.write('\t'.join(str(w) for w in info) + '\n')

