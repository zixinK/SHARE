import json
# set_display_options()
import os
import re
import subprocess
import logging
import zipfile
import tarfile
import urllib.request as request

from typing import List, Dict

import pandas as pd

from Preprocessing.config_global import global_config
from util.print_util import iprint
from util.misc_util import tokens_to_spans

logger = logging.getLogger(__name__)
data_sets = [
    {#Check
		'name': 'movies',
		'url': 'https://www.eraserbenchmark.com/zipped/movies.tar.gz',
	},
    # {
    #     'name': 'vitaminc_rationale',
    #     'url': 'https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc_rationale.zip'
    # }
]

sample_objects_only=False

sets = [('test', 'test.jsonl'),
		('train', 'train.jsonl'),
		('dev', 'val.jsonl'),
		]

dtr = f'{global_config["data_directory"]}'

def download_and_extract(data_sets=data_sets, data_dir=dtr):
	iprint(f'Downloading and processing eraser datasets to {data_dir}')
	object_sample = []
	for dataset in data_sets:
		extracted_dir = look_for_extracted_dir(data_dir, dataset['name'])
		if extracted_dir:
			iprint(f'Extracted files already exist at {extracted_dir}')
		else:
			iprint(f'Processing {dataset["name"]}...')
			dataset_dir = os.path.join(data_dir, dataset['name'])
			data_file = "%s.zip" % dataset['name']
			request.urlretrieve(dataset['url'], data_file)
			with tarfile.open(data_file) as tar_ref:
				tar_ref.extractall(data_dir)
			doc_df = read_documents(extracted_dir)
			iprint('Document sample:')
			iprint(doc_df.head(5))

			iprint("Completed! Stored at %s" % data_dir)

		doc_df = read_documents(extracted_dir)
		# 只有两列 一列 document 一列class
		iprint('Document sample:')
		iprint(doc_df.head(5))

		for setnum, (setname, setfile) in enumerate(sets):
			iprint(f'{setname} set:')
			set_objects = read_jsonlines(os.path.join(extracted_dir, setfile))

			if sample_objects_only:
				sample_object = set_objects[0]
				sample_object['dataset'] = dataset
				sample_object['set'] = setname
				object_sample.append(sample_object)
				break

			set_objects = [process_set_object(set_object, doc_df, dataset["name"]) for set_object in set_objects]
			set_objects = [object for object in set_objects if object is not None]

			set_df = pd.DataFrame(set_objects, columns=['annotation_id', 'classification',
														'document',
														'document_rationale_values',
														'query']).rename(
				columns={'annotation_id': 'id',
						 'classification': 'label',
						 'document_rationale_values':'ground-truth_rationale'})
			iprint('Set sample:')
			iprint(set_df.head(5))

			# iprint('Evidences sample')
			# iprint(pformat(set_df.iloc[0]['evidences']))
			set_filepath = os.path.join(extracted_dir, f'{setname}.json')
			iprint(f'Writing to {set_filepath}')
			set_df.to_json(set_filepath, orient='records', lines=True)

			sample_filepath = os.path.join(extracted_dir, f'{setname}_sample.html')
			# sample_and_output_as_html(output_df=set_df,
			# 						  output_path=sample_filepath,
			# 						  sample_function=lambda df: df.sample(n=min(100,set_df.shape[0]), random_state=seed),
			# 						  text_span_value_columns={'document':{'document_rationale_spans':['document_rationale_values']},
			# 												   'query':{'query_rationale_spans':['query_rationale_values']}},
			# 						  scale_weird_ranges=False)

			iprint(f'Done with {setname} set')

		pass

		if sample_objects_only:
			write_jsonlines(object_sample, os.path.join(data_dir, 'object_sample.json'))

		iprint('Done!')

def look_for_extracted_dir(dataset_dir, extracted_filename):
	if os.path.exists(os.path.join(dataset_dir, 'data', extracted_filename)):
		extracted_dir = os.path.join(dataset_dir, 'data', extracted_filename)
	elif os.path.exists(os.path.join(dataset_dir, extracted_filename)):
		extracted_dir = os.path.join(dataset_dir, extracted_filename)
	else:
		extracted_dir = None
	return extracted_dir

def read_documents(doc_dir):
	iprint(f'Reading documents from {doc_dir}')
	iprint(os.listdir(doc_dir))
	docfilename = [filename for filename in os.listdir(doc_dir) if filename in ['docs', 'docs.jsonl']][0]
	doc_objs = read_jsonlines_or_dir(os.path.join(doc_dir, docfilename))
	doc_df = pd.DataFrame(doc_objs)
	doc_df.set_index('docid', inplace=True)
	return doc_df

def read_jsonlines_or_dir(path):
	if os.path.isdir(path):
		return read_json_dir(path)
	else:
		return read_jsonlines(path)

doc_fn_patterns = [
	re.compile("(?P<docid>.+)"),
	# re.compile("(?P<class>[a-z]+)R_(?P<docid>[0-9]+)\.txt")
]

#

def read_json_dir(dirpath):
	iprint('Parsing doc dir {}'.format(dirpath))

	objs = []
	filenames = os.listdir(dirpath)
	for i, filename in enumerate(filenames):
		filepath = os.path.join(dirpath, filename)
		with open(filepath, 'r',encoding = 'utf-8') as f:

			if filepath.endswith('.json'):
				obj = json.load(f)
			else:
				matched = False
				for pattern in doc_fn_patterns:
					m = re.match(pattern, filename)
					if m:
						matched = True
						obj = {'document': f.read()}
						obj.update(m.groupdict())
						objs.append(obj)
						if i == 0:
							iprint('{} --> {}'.format(filename, m.groupdict()), 1)
						break

				if not matched:
					raise Exception('Script does not know how to read file {}'.format(filepath))

	iprint('{} items loaded.'.format(len(objs)), 1)
	return objs

def read_jsonlines(filepath):
	iprint('Parsing jsonl file {}'.format(filepath))
	with open(filepath, 'r') as f:
		objs = [json.loads(line) for line in f.readlines()]
	iprint('{} items loaded.'.format(len(objs)), 1)
	return objs


def process_set_object(set_object, doc_df, dataset_name):

	if len(set_object['evidences']) == 0: #Movies dataset has 1 or 2 empty examples
		return None

	# set_object['evidences'] is either a list of dictionaries nested in a 1-element list, or a list of 1-element lists with one evidence dict each.
	evidences = [evidence for sublist in set_object['evidences'] for evidence in sublist]


	if set_object.get('docids') is None:
		evidence_docids = list(set([evidence['docid'] for evidence in evidences]))
		assert len(evidence_docids) == 1
		document_id = evidence_docids[0]
	else:
		assert len(set_object['docids']) == 1
		document_id = set_object['docids'][0]

	set_object['document'] = pre_document(doc_df.loc[document_id]['document'])


	doc_evidences = [evidence for evidence in evidences if evidence['docid'] == document_id]



	set_object['document_rationale_values'] = evidences_to_rationale(set_object['document'], doc_evidences)
	# assert set_object['document_rationale_spans'] is not None #We should never have a missing document rationale



	# pprint(set_object)

	# set_object['document'] = string_or_list_to_list(doc_df.loc[document_id]['document'])

	return set_object


# def string_or_list_to_list(input_str):
# 	'''
#     字符串转列表,或者列表list转列表list
#     :param input_str: 输入的内容，可以是一个字符串，也可以是一个list
#     :return: 返回list
#     '''
# 	# print('-' * 100)
# 	if isinstance(input_str, list):
# 		# print('---list to list')
# 		output_list = input_str
# 	elif isinstance(input_str, str):
# 		# print('---str to list')
# 		input_str = str(input_str).strip(' ').strip("'").strip('"').strip(',').strip('，')
# 		output_list = input_str.split(' ')
# 	# output_list = ",".join(input_str)
# 	else:
# 		# print('---else to list')
# 		input_str = str(input_str).strip('[').strip(']').strip('"').strip("'").strip('"').split(',')  # 去掉多余的字符串
# 		output_list = ",".join(input_str)
#
# 	# print('input_str={},output_list={}'.format(input_str, output_list))
# 	return output_list

def pre_document(do):
    do = re.sub(r"\.\.", " .. ", do)
    do = re.sub(r"\|\|", " | | ", do)
    do = re.sub(r"\-\-", " - - ", do)
    do = re.sub(r"\.", " . ", do)
    do = re.sub(r"\-", " - ", do)
    do = re.sub(r"\n", " ", do)
    do = re.sub(r"\/", " / ", do)
    do = re.sub(r"\=", " = ", do)
    do = re.sub(r"\+", " + ", do)
    do = re.sub(r"\&", " & ", do)
    do = re.sub(r"\_", " _ ", do)
    do = re.sub(r"\~", " ~ ", do)
    do = re.sub(r"\@", " @ ", do)
    do = re.sub(r"\$", " $ ", do)
    do = re.sub(r"\`", " ` ", do)
    do = re.sub(r"\^", " ^ ", do)
    do = re.sub(r"\{", " { ", do)
    do = re.sub(r"\  ", " ", do)
    do = re.sub(r'\"', ' " ', do)
    do = re.sub(r'\#', ' # ', do)
    do = re.sub(r"\\", " \ ", do)

    return do

def evidences_to_rationale(text, evidences, test=True):
	if evidences is None or len(evidences) == 0:
		return None, None
	tokens = text.split()

	spans = tokens_to_spans(tokens, text)

	values = [0.0 for token in tokens]
	for evidence in evidences:
		evidence['text'] = pre_document(evidence['text'])
		if test:
			spantext = text[spans[evidence['start_token']][0]:spans[evidence['end_token'] - 1][1]].replace('\n', ' ')
			t_len = len(evidence['text'].split())
			t_n = min(20, t_len)
			if not spantext.split() == evidence['text'].split():
				for i in range(len(tokens) - t_n + 1):
					if tokens[i:i + t_n] == evidence['text'].split()[0:t_n]:
						evidence['start_token'] = i
						# iprint(f'"{tokens[i:i+t_n]}"')
						# iprint(f'"{evidence["text"].split()[0:t_n]}"')
						break
				evidence['end_token'] = evidence['start_token'] + t_len
				spantext = text[spans[evidence['start_token']][0]:spans[evidence['end_token'] - 1][1]].replace('\n',
																											   ' ')

			if not spantext.split() == evidence['text'].split():
				iprint(f'Mismatch: "{spantext.split()}" vs "{evidence["text"].split()}"')
				iprint(evidence['docid'])

		for i in range(evidence['start_token'], evidence['end_token']):
			values[i] = 1.0
	values = [i for i in range(len(values)) if values[i] == 1]

	# if len(evidences) > 1 and any([evidence['end_sentence'] > -1 for evidence in evidences]):
	# 	iprint(pformat(evidences))
	# 	x=1

	return values

def write_jsonlines(obj_list:List[Dict], filepath:str):
	iprint(f'Dumping list of {len(obj_list)} objects to {filepath}')
	with open(filepath,'w') as f:
		f.writelines(json.dumps(obj) for obj in obj_list)
	iprint('Done')

