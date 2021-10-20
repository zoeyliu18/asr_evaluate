import io, os, argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input path')
	
	args = parser.parse_args()

	for folder in os.listdir(args.input):
		if os.path.isdir(args.input + folder) is True:
			for file in os.listdir(args.input + folder + '/'):
				if file.startswith('16k'):
					file_name = file.split('.')[0]
					file_name = file_name.split('_')
					hour = file_name[2][0 : 2]
					date = file_name[-2]
					new_name = 'SWH-' + hour + '-' + date + '.wav'
					os.rename(args.input + folder + '/' + file, args.input + folder + '/' + new_name)


