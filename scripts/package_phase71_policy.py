"""Pack a complete aggregate into at most four bounded lossless DAG parts."""
import gzip,hashlib,importlib.util,json,pathlib,sys


def main():
    root=pathlib.Path.cwd();base=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2])
    spec=importlib.util.spec_from_file_location('codec',root/'docs/_ext/wiki_phase69_efficiencies.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    spec=importlib.util.spec_from_file_location('native',root/'docs/_ext/wiki_phase69.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    value=native.public_pid_labels(json.loads((base/'aggregate.json').read_text()))
    data=codec.canonical(value);encoded=codec.pack(value)
    assert codec.canonical(codec.unpack(encoded))==data
    parts=[];current=[];size=100
    for node in encoded['nodes']:
     cost=len(codec.canonical(node))
     if size+cost>4*1024*1024:
      parts.append(current);current=[];size=100
     current.append(node);size+=cost
    if current:parts.append(current)
    assert len(parts)<=4,'Supplement exceeds fixed four-part publication budget'
    dest.mkdir(parents=True,exist_ok=False)
    binding={'version':'phase71-policy-dag-binding-v1','decoded_sha256':hashlib.sha256(data).hexdigest(),
             'root':encoded['root'],'nodes':len(encoded['nodes']),'parts':[],'decoded_bytes':len(data)}
    for i,nodes in enumerate(parts):
     part=codec.canonical({'encoding':'typed-json-dag-part-v1','part':i,'nodes':nodes})
     assert len(part)<=4*1024*1024
     filename=f'aggregates-{i}.json.gz';path=dest/filename
     path.write_bytes(gzip.compress(part,mtime=0))
     binding['parts'].append({'source':filename,'filename':f'phase71-policy-part-{i}.json','bytes':len(part),'sha256':hashlib.sha256(part).hexdigest()})
    (dest/'binding.json').write_bytes(codec.canonical(binding))
    (base/'control/publication-package.json').write_bytes(codec.canonical(binding))
    print(json.dumps({'decoded_bytes':len(data),'parts':len(parts),'encoded_bytes':sum(p['bytes'] for p in binding['parts']),'nodes':len(encoded['nodes'])}))


if __name__ == '__main__':
    main()
