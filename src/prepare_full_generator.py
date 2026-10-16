"""Prepare readable private source/data transport for uncapped CPU training."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from src.data.prepare_fairytale import sha


def prepare(data,protocol,output,dataset,action,kernel_source=None,epoch=0,run_tag=None,validation_source=None,task="generator"):
    if task not in ("generator", "reranker"):raise ValueError("Unknown full-data task")
    if action not in ("smoke","train","validate"):raise ValueError("Invalid full-data action")
    if action=="validate" and not kernel_source:raise ValueError("Validation needs an immutable training kernel source")
    if validation_source and action!="validate":raise ValueError("Validation shards apply only to validation")
    if type(epoch) is not int or not 0<=epoch<=3:raise ValueError("Expected a declared epoch snapshot")
    if run_tag is not None and (not run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in run_tag)):
        raise ValueError("Use a plain lowercase versioned run tag")
    output=Path(output);data=Path(data)
    if output.exists():raise ValueError("Preserve prepared payloads; choose a fresh output")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("Commit the coherent source/protocol milestone before packaging")
    paths=["src/__init__.py","src/data/__init__.py","src/data/prepare_fairytale.py",
           "src/narrative_training.py","src/full_generator_training.py","src/resumable_training.py"]
    if task=="reranker":paths += ["src/reranker_training.py", "src/full_reranker_training.py"]
    files={name:Path(name).read_bytes() for name in paths}
    files["docs/full-coverage-protocol.json"]=Path(protocol).read_bytes()
    for path in data.rglob("*"):
        if path.is_file() and "test" not in path.relative_to(data).parts:
            files["private-data/"+path.relative_to(data).as_posix()]=path.read_bytes()
    manifest={"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "files":{name:sha(raw) for name,raw in files.items()},"action":action,
              "test_QA_packaged":False,"plain_source":True,"task":task}
    raw=json.dumps(manifest,indent=2).encode()
    files["full-payload-provenance.json"]=raw
    output.mkdir(parents=True);upload=output/"dataset-upload";upload.mkdir()
    with zipfile.ZipFile(upload/"payload.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():archive.writestr(name,content)
    license_path=next(Path("data/external/fairytaleqa").glob("FairytaleQAData-*/LICENSE"))
    (upload/"LICENSE").write_bytes(license_path.read_bytes())
    (upload/"dataset-metadata.json").write_text(json.dumps({"id":dataset,
        "title":"Private full narrative source v6","licenses":[{"name":"other"}],
        "description":"All custom-family train/validation FairytaleQA under Apache2; readable code; no test QA."},indent=2),encoding="utf-8")
    runner=Path("src/cloud/full_generator_runner.py" if task=="generator" else "src/cloud/full_reranker_runner.py").read_text(encoding="utf-8")
    runner=runner.replace("EXPECTED_MANIFEST_SHA256 = None","EXPECTED_MANIFEST_SHA256 = "+repr(sha(raw)))
    runner=runner.replace('ACTION = "smoke"',"ACTION = "+repr(action))
    runner=runner.replace('USE_PREVIOUS_RUN = False',"USE_PREVIOUS_RUN = "+repr(bool(kernel_source)))
    runner=runner.replace('VALIDATION_EPOCH = 0',"VALIDATION_EPOCH = "+str(epoch))
    runner=runner.replace('USE_PREVIOUS_VALIDATION = False',"USE_PREVIOUS_VALIDATION = "+repr(bool(validation_source)))
    (output/"run.py").write_text(runner,encoding="utf-8",newline="\n")
    slug="portfolio-narrative-full-"+("resume" if task=="generator" else "reranker")+"-"+action+"-v6"
    if action=="validate":slug+="-epoch"+str(epoch)
    if run_tag:slug+="-"+run_tag
    (output/"kernel-metadata.json").write_text(json.dumps({"id":"aikyatansinha/"+slug,
        "title":slug.replace("-"," ").title(),"code_file":"run.py","language":"python","kernel_type":"script",
        "is_private":True,"enable_gpu":False,"enable_internet":True,"dataset_sources":[dataset],
        "competition_sources":[],"kernel_sources":([kernel_source] if kernel_source else [])+
        ([validation_source] if validation_source else [])},indent=2),encoding="utf-8")
    receipt={**manifest,"payload_zip_sha256":sha((upload/"payload.zip").read_bytes()),
             "runner_sha256":sha((output/"run.py").read_bytes()),"dataset":dataset,
             "kernel_source":kernel_source,"validation_source":validation_source,"validation_epoch":epoch,"run_tag":run_tag}
    (output/"preflight.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({"kernel":"aikyatansinha/"+slug,"action":action,"test_QA_packaged":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","protocol","output","dataset"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--action",choices=("smoke","train","validate"),default="smoke")
    parser.add_argument("--kernel-source");parser.add_argument("--epoch",type=int,default=0)
    parser.add_argument("--run-tag")
    parser.add_argument("--validation-source")
    parser.add_argument("--task",choices=("generator","reranker"),default="generator")
    a=parser.parse_args();prepare(a.data,a.protocol,a.output,a.dataset,a.action,a.kernel_source,a.epoch,a.run_tag,a.validation_source,a.task)
