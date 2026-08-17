import urllib.request, urllib.parse, json

headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.github.v3+json'}

for folder in ['jarvis', 'ok_jarvis', 'hey_kitt', 'computer', 'alfred', 'TARS']:
    url = ('https://api.github.com/repos/fwartner/home-assistant-wakewords-collection'
           '/contents/en/' + urllib.parse.quote(folder))
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            items = json.loads(r.read())
        print('[' + folder + ']')
        for item in items:
            size_kb = item.get('size', 0) // 1024
            dl = item.get('download_url') or ''
            print('  ' + item['name'] + ' (' + str(size_kb) + ' KB)')
            if dl and item['name'].endswith('.onnx'):
                print('  download: ' + dl)
    except Exception as e:
        print('[' + folder + '] Error: ' + str(e))
