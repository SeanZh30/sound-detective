"""Download the 12 original ESC-50 WAV files listed in dataset.json."""
import json, pathlib, urllib.request
root=pathlib.Path(__file__).resolve().parents[1]
for item in json.loads((root/'src/dataset.json').read_text()):
    target=root/'public'/item['file']
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/'+target.name,target)
    print(target.name, target.stat().st_size)
