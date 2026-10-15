"""Prepare a private bounded reader experiment; never publish literary payloads."""
import ast
import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(folder=Path('data/interim/kaggle-reader-v3')):
    from src.data.make_dataset import strip_gutenberg, write_processed
    from src.features.build_features import chunk_book
    from src.models.train_model import HybridIndex, DENSE_REVISION
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path('docs/v3-prespecified-protocol.json').read_text())
    for specification in protocol['models']:
        metadata = json.load(urllib.request.urlopen(
            f"https://huggingface.co/api/models/{specification['name']}/revision/{specification['revision']}", timeout=30))
        if metadata['sha'] != specification['revision'] or metadata.get('gated') is not False:
            raise ValueError('Model revision or ungated eligibility changed')
        if metadata.get('cardData', {}).get('license') != 'cc-by-4.0':
            raise ValueError('Expected model-card license changed')
        card = urllib.request.urlopen(specification['source'], timeout=30).read()
        specification['model_card_sha256'] = digest(card)
    previous = json.loads(Path('docs/v2-validation-selection.json').read_text())
    qa_path = Path('data/external/narrativeqa/qaps.csv')
    if digest(qa_path.read_bytes()) != previous['source']['qa_csv_sha256']:
        raise ValueError('Pinned QA source changed')
    documents = {r['document_id']: r for r in csv.DictReader(
        Path('data/external/narrativeqa/documents.csv').open(encoding='utf-8'))}
    document = documents[protocol['fresh_test_document']]
    if document['set'] != 'test' or '/1063.txt.utf-8' not in document['story_url']:
        raise ValueError('Prespecified fresh source mapping mismatch')
    selection = previous['selected']['config']
    if selection != {'name': 'fine96_context_top12', 'words': 96, 'overlap': 48,
                     'k': 12, 'expand': 700, 'sliding': True}:
        raise ValueError('Expected frozen v2 retrieval configuration')
    controls_ast = ast.parse(Path('src/reader_evaluation.py').read_text())
    controls = next(ast.literal_eval(node.value)['valid'] for node in controls_ast.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == 'CONTROLS' for t in node.targets))
    dorian = Path('data/processed/dorian.txt').read_text(encoding='utf-8')
    validation, gold, fresh_inputs, fresh_gold = [], [], [], []
    for number, row in enumerate(csv.DictReader(qa_path.open(encoding='utf-8'))):
        identifier = f'narrativeqa:{number}'
        if row['document_id'] == '879287f285743b08d86c3bd0831ec1e49fa92e39':
            if row['set'] != 'valid':
                raise ValueError('Validation split mismatch')
            saved = json.loads(Path(f'data/interim/v2-fine96_context_top12-narrativeqa-{number}.json').read_text())
            if saved['source'] != previous['source'] or saved['config'] != selection:
                raise ValueError('Frozen context provenance mismatch')
            passages = saved['result']['passages']
            validation.append({'id': identifier, 'question': row['question'], 'passages': passages})
            gold.append({'id': identifier, 'kind': 'validation', 'answers': [row['answer1'], row['answer2']]})
        elif row['document_id'] == protocol['fresh_test_document']:
            # Package without printing or using fresh QA for any policy decision.
            fresh_inputs.append({'id': identifier, 'question': row['question']})
            fresh_gold.append({'id': identifier, 'answers': [row['answer1'], row['answer2']]})
    for number, question in enumerate(controls):
        identifier = f'v2-control-{number}'
        saved = json.loads(Path(f'data/interim/v2-fine96_context_top12-{identifier}.json').read_text())
        validation.append({'id': identifier, 'question': question, 'passages': saved['result']['passages']})
        gold.append({'id': identifier, 'kind': 'control', 'answers': ['']})
    for row in validation:
        for passage in row['passages']:
            if passage['book'] != 'dorian' or dorian[passage['start']:passage['end']] != passage['text']:
                raise ValueError('Validation passage/source mismatch')
    url = 'https://www.gutenberg.org/ebooks/1063.txt.utf-8'
    raw = urllib.request.urlopen(url, timeout=45).read()
    Path('data/raw/cask.txt').write_bytes(raw)
    write_processed(Path('data/processed/cask.txt'), strip_gutenberg(raw.decode('utf-8-sig')))
    text = Path('data/processed/cask.txt').read_text(encoding='utf-8')
    passages = chunk_book('cask', text, 96, 48)
    index = HybridIndex(passages, backend='sentence-transformer', model_revision=DENSE_REVISION)
    index.source_hashes = {'cask': digest(text.encode())}
    index.chunk_contract = {'words': 96, 'overlap': 48}
    index.verify_sources(Path('data/processed'))
    index.verify_chunk_contract(Path('data/processed'), 96, 48)
    index.save(Path('models/v3-cask.lrag'))
    files = {'inputs.json': json.dumps({'rows': validation}).encode(),
             'gold.json': json.dumps(gold).encode(),
             'fresh-inputs.json': json.dumps(fresh_inputs).encode(),
             'fresh-gold.json': json.dumps(fresh_gold).encode(),
             'fresh.lrag': Path('models/v3-cask.lrag').read_bytes(),
             'processed/cask.txt': Path('data/processed/cask.txt').read_bytes()}
    protocol.update({'input_sha256': digest(files['inputs.json']),
                     'gold_sha256': digest(files['gold.json']),
                     'fresh_inputs_sha256': digest(files['fresh-inputs.json']),
                     'fresh_gold_sha256': digest(files['fresh-gold.json']),
                     'fresh_index_sha256': digest(files['fresh.lrag']),
                     'dorian_source_sha256': digest(dorian.encode()),
                     'fresh_source': {'url': url, 'raw_sha256': digest(raw),
                                      'normalized_text_sha256': digest(text.encode()),
                                      'passages': len(passages), 'words': len(text.split())},
                     'validation_questions': 29, 'authored_controls': 4,
                     'fresh_questions': len(fresh_inputs),
                     'qa_csv_sha256': digest(qa_path.read_bytes()),
                     'qa_source_commit': '904246f6d1fe99a99a08a03501fe3e619af2cee5',
                     'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     'reader_runner_sha256': digest(Path('src/reader_cloud.py').read_bytes())})
    files['protocol.json'] = json.dumps(protocol, indent=2).encode()
    # Package only the source modules needed for the exact frozen hybrid policy.
    source_files = ['src/__init__.py', 'src/features/__init__.py', 'src/features/build_features.py',
                    'src/models/__init__.py', 'src/models/train_model.py', 'src/models/grounded_spans.py',
                    'src/models/loading.py']
    for path in source_files:
        files[path] = Path(path).read_bytes()
    files['reader_cloud.py'] = Path('src/reader_cloud.py').read_bytes()
    packed = {name: base64.b64encode(gzip.compress(raw)).decode() for name, raw in files.items()}
    bootstrap = '''import os,sys,time,threading,subprocess,json,base64,gzip
from pathlib import Path
os.environ['READER_STARTED_MONOTONIC']=str(time.monotonic())
watchdog=threading.Timer(900,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','sentence-transformers==5.1.2'],check=True,timeout=180)
folder=Path('/kaggle/working/private-input');folder.mkdir(parents=True,exist_ok=True)
packed=PACKED_PAYLOAD
for name,raw in packed.items():
 path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gzip.decompress(base64.b64decode(raw)))
sys.path.insert(0,str(folder))
namespace={'__name__':'reader_cloud'}
exec(compile((folder/'reader_cloud.py').read_text(),'reader_cloud.py','exec'),namespace)
namespace['run'](folder,Path('/kaggle/working'),900)
os._exit(0)
'''.replace('PACKED_PAYLOAD', repr(packed))
    (folder / 'run.py').write_text(bootstrap, encoding='utf-8')
    metadata = {'id': 'aikyatansinha/portfolio-literary-reader-v3',
                'title': 'Portfolio Literary Reader V3', 'code_file': 'run.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = {'status': 'Prepared; GPU launch awaits root coordination and smoke checks',
              'protocol': protocol, 'private_bootstrap_sha256': digest((folder / 'run.py').read_bytes()),
              'private_bootstrap_bytes': (folder / 'run.py').stat().st_size,
              'validation_exact_source_slices': sum(len(r['passages']) for r in validation),
              'fresh_gold_available_to_forward_inference': False,
              'raw_literary_payloads_public': False}
    Path('docs/v3-preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'validation_rows': len(validation), 'fresh_questions': len(fresh_inputs),
                      'fresh_passages': len(passages), 'bootstrap_bytes': report['private_bootstrap_bytes']}))
    return report
