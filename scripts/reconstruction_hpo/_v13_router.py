"""Private, file-backed v13 production router (auth-false)."""
from __future__ import annotations
import hashlib, os, stat, time, weakref
from pathlib import Path
from typing import Any, Iterable, Mapping
from . import _v13_core as _core

_NODES=("TerminalNode.v5","PairNode.v5","LocatorNode.v5","PilotNode.v5","LadderNode.v5","PointerNode.v5","HpoNode.v5","FinalNode.v5")
_REF=("path","byte_length","sha256","media_type","schema","version")
_NATIVE={"SchedulerNativeReceipt.v11","ControllerNativeReceipt.v11","RuntimeNativeReceipt.v11","TelemetryNativeReceipt.v11","CheckpointNativeReceipt.v11","ValidationNativeReceipt.v11","MidpointNativeReceipt.v11","ResourceNativeReceipt.v11"}
_MAX_AGE_NS=300_000_000_000

def _keys(v:Mapping[str,Any], expected:Iterable[str], label:str)->None:
    wanted=tuple(expected)
    if not isinstance(v,Mapping) or len(v)!=len(wanted) or set(v)!=set(wanted): raise _core.VerificationError(f"E_KEYS:{label}")

def _reject_none(v:Any,label:str)->None:
    if v is None: raise _core.VerificationError(f"E_TYPE:{label}:None")
    if isinstance(v,Mapping):
        for k,x in v.items(): _reject_none(x,f"{label}.{k}")
    elif isinstance(v,list):
        for i,x in enumerate(v): _reject_none(x,f"{label}[{i}]")

def _root(path:Path, roots:tuple[Path,...])->Path:
    matches=[r for r in roots if path.is_relative_to(r)]
    if not matches: raise _core.VerificationError("E_PATH_ROOT")
    root=max(matches,key=lambda r:len(r.parts)); st=root.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode): raise _core.VerificationError("E_PATH_ROOT")
    if hasattr(os,"getuid") and st.st_uid!=os.getuid(): raise _core.VerificationError("E_PATH_ROOT_OWNER")
    if st.st_mode&(stat.S_IWGRP|stat.S_IWOTH): raise _core.VerificationError("E_PATH_ROOT_MODE")
    return root

def _stable(ref:Mapping[str,Any], trusted:Iterable[str])->bytes:
    _keys(ref,_REF,"ArtifactRef.v5"); _core.require_hex(ref["sha256"],"ArtifactRef sha256")
    if isinstance(ref["byte_length"],bool) or not isinstance(ref["byte_length"],int) or ref["byte_length"]<=0: raise _core.VerificationError("E_RAW_LENGTH")
    path=Path(ref["path"])
    if not path.is_absolute() or path!=Path(os.path.normpath(str(path))): raise _core.VerificationError("E_PATH_ABSOLUTE")
    try: roots=tuple(Path(x).resolve(strict=True) for x in trusted)
    except (FileNotFoundError,NotADirectoryError,OSError) as exc: raise _core.VerificationError("E_PATH_ROOT") from exc
    root=_root(path,roots); cur=root
    for part in path.relative_to(root).parts:
        cur/=part; st=cur.lstat()
        if stat.S_ISLNK(st.st_mode): raise _core.VerificationError("E_PATH_TYPE")
        if hasattr(os,"getuid") and st.st_uid!=os.getuid(): raise _core.VerificationError("E_PATH_OWNER")
        if st.st_mode&(stat.S_IWGRP|stat.S_IWOTH): raise _core.VerificationError("E_PATH_MODE")
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode): raise _core.VerificationError("E_PATH_TYPE")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        opened=os.fstat(fd); ident=(opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns)
        if ident!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns): raise _core.VerificationError("E_PATH_CHANGED")
        chunks=[]
        while True:
            chunk=os.read(fd,1<<20)
            if not chunk: break
            chunks.append(chunk)
        after=os.fstat(fd)
        if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=ident: raise _core.VerificationError("E_PATH_CHANGED")
    finally: os.close(fd)
    raw=b"".join(chunks)
    if len(raw)!=ref["byte_length"] or hashlib.sha256(raw).hexdigest()!=ref["sha256"]: raise _core.VerificationError("E_RAW_HASH")
    return raw

def _digest(v:Mapping[str,Any])->str: return hashlib.sha256(_core.jcs({k:x for k,x in v.items() if k!="digest"})).hexdigest()
def _auth(v:Any)->None:
    if not isinstance(v,Mapping) or not v or any(x is not False for x in v.values()): raise _core.VerificationError("E_AUTHORIZATION")
def _refs(v:Any):
    if isinstance(v,Mapping):
        if tuple(v)==_REF: yield v
        else:
            for x in v.values(): yield from _refs(x)
    elif isinstance(v,list):
        for x in v: yield from _refs(x)

def _leaf(ref:Mapping[str,Any],roots:tuple[str,...],spec:Mapping[str,Any])->None:
    raw=_stable(ref,roots)
    if not isinstance(ref["media_type"],str): raise _core.VerificationError("E_API_TYPE")
    if "json" not in ref["media_type"]: return
    value=_core.parse_json_bytes(raw)
    if not isinstance(value,Mapping): raise _core.VerificationError("E_API_TYPE")
    name=ref["schema"]; _reject_none(value,name)
    if name=="NormalizedTensorManifest.v12": _core.validate_normalized_manifest(spec,value)
    elif name=="StateIntegrityReceipt.v12": _core.validate_state_integrity(spec,value)
    elif name=="ABBAEvaluationReceipt.v11": _core.validate_abba(spec,value)
    elif name in _NATIVE: _core.validate_native_receipt(name,spec,value)
    elif name in spec["schemas"]: _core.validate_common_receipt(spec,value,name)
    for nested in _refs(value): _leaf(nested,roots,spec)

def _make_api():
    seal=object(); live:weakref.WeakKeyDictionary[object,tuple[int,int,str]]=weakref.WeakKeyDictionary()
    class VerifiedChain:
        __slots__=("_pid","_digest","_seal","__weakref__")
        def __new__(cls,issuer:object,pid:int,digest:str):
            if issuer is not seal: raise TypeError("nonconstructible capability")
            return super().__new__(cls)
        def __init__(self,issuer:object,pid:int,digest:str): object.__setattr__(self,"_pid",pid); object.__setattr__(self,"_digest",digest); object.__setattr__(self,"_seal",issuer)
        def __setattr__(self,n:str,v:object)->None: raise AttributeError("immutable capability")
    def verify_nodes(repo:str,refs:tuple[Mapping[str,Any],...],roots:tuple[str,...])->object:
        if len(refs)!=8: raise TypeError("exactly eight node ArtifactRefs are required")
        spec=_core.load_spec(repo); previous=None; digests=[]
        for i,(ref,name) in enumerate(zip(refs,_NODES)):
            node=_core.parse_json_bytes(_stable(ref,roots))
            if not isinstance(node,Mapping): raise _core.VerificationError("E_API_TYPE")
            schema=spec["schemas"][name]; _keys(node,schema["exact_keys"],name)
            for key,expected in schema.get("required",{}).items():
                if node[key]!=expected: raise _core.VerificationError(f"E_STAGE_ORDER:{name}.{key}")
            _auth(node["authorization"]); digest=_digest(node)
            if node["digest"]!=digest: raise _core.VerificationError("E_NODE_DIGEST")
            if i and node["previous_digest"]!=previous: raise _core.VerificationError("E_PREVIOUS_DIGEST")
            for leaf in _refs(node): _leaf(leaf,roots,spec)
            previous=digest; digests.append(digest)
            if i==1:
                gate=spec["downstream_fail_closed_gates"]["requirements"][0]
                if gate["id"]!="F01" or gate["state"]!="MISSING_OR_NOT_ACCEPTED" or gate["failure_code"]!="E_FEASIBILITY_SHAPE_AUTHORITY": raise _core.VerificationError("E_AUTHORITY_OVERRIDE_FORBIDDEN")
                raise _core.VerificationError("E_FEASIBILITY_SHAPE_AUTHORITY")
        digest=hashlib.sha256("".join(digests).encode()).hexdigest(); cap=VerifiedChain(seal,os.getpid(),digest); live[cap]=(os.getpid(),time.monotonic_ns()+_MAX_AGE_NS,digest); return cap
    def decide(cap:object)->str:
        if type(cap) is not VerifiedChain: raise _core.VerificationError("E_API_CAPABILITY")
        rec=live.pop(cap,None)
        if rec is None or cap._seal is not seal or cap._pid!=os.getpid() or rec[0]!=os.getpid() or time.monotonic_ns()>rec[1] or cap._digest!=rec[2]: raise _core.VerificationError("E_API_CAPABILITY")
        return "E_FEASIBILITY_SHAPE_AUTHORITY"
    return verify_nodes,decide

verify_nodes,decide=_make_api(); del _make_api
__all__:list[str]=[]
