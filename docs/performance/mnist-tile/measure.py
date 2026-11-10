# Run from the backend repository: python3 docs/performance/mnist-tile/measure.py ARTIFACT_DIR SCENARIO...
from pathlib import Path
import subprocess,os,json,re,statistics,sys,time
root=Path(sys.argv[1]).resolve()
backend=Path(__file__).resolve().parents[3]
onnx=backend.parent/'onnx'
env=dict(os.environ,GOMAXPROCS='1',CGO_ENABLED='0')
scenarios={
'mnist-kernel':('kernel','^BenchmarkConv2DInto(NoBias)?$/^MNIST',7,'700ms',backend),
'kernels':('kernel','^BenchmarkConv2DInto$',5,'200ms',backend),
'purego-kernels':('kernel-purego','^BenchmarkConv2DInto$',5,'200ms',backend),
'mnist':('onnx','^BenchmarkMNIST$',7,'700ms',onnx),
'nodes':('onnx','^BenchmarkMNISTNodes$/(00_Conv|04_Conv)$',5,'400ms',onnx),
'purego':('onnx-purego','^BenchmarkMNIST$',5,'500ms',onnx),
}
for scenario in sys.argv[2:]:
    prefix,pattern,count,duration,cwd=scenarios[scenario]
    runs=[]
    for repeat in range(count):
        for variant in (['before','after'] if repeat%2==0 else ['after','before']):
            command=[str(root/f'{prefix}-{variant}.test'),'-test.run=^$','-test.bench='+pattern,'-test.benchtime='+duration,'-test.benchmem']
            result=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True,check=True)
            with (root/f'{scenario}-{variant}.txt').open('a') as stream:
                stream.write(f'\n# Repetition {repeat+1}; {variant}\n'+result.stdout)
            runs.append({'repeat':repeat+1,'variant':variant,'command':command,'cwd':str(cwd),'stdout':result.stdout})
            print(scenario,repeat+1,variant,flush=True)
    (root/(scenario+'-runs.json')).write_text(json.dumps(runs,indent=2))
    samples={}
    for run in runs:
        for name,ns,allocated,allocs in re.findall(r'^(Benchmark\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op',run['stdout'],re.M):
            samples.setdefault(name,{}).setdefault(run['variant'],[]).append({'ns':float(ns),'bytes':int(allocated),'allocs':int(allocs)})
    summary={}
    for name,variants in samples.items():
        summary[name]={}
        for variant,sample in variants.items():
            values=[x['ns'] for x in sample]
            summary[name][variant]={'median_ns':statistics.median(values),'minimum_ns':min(values),'maximum_ns':max(values),'bytes':sorted(set(x['bytes'] for x in sample)),'allocs':sorted(set(x['allocs'] for x in sample)),'samples':sample}
        b=summary[name]['before']['median_ns']; a=summary[name]['after']['median_ns']
        summary[name]['change_percent']=100*(a/b-1)
        print(f'{name:48} {b:10.1f} -> {a:10.1f} {100*(a/b-1):+6.2f}%',flush=True)
    (root/(scenario+'-summary.json')).write_text(json.dumps(summary,indent=2))
